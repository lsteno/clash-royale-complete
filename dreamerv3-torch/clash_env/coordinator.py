import time
import numpy as np
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from .pixel_env import PixelClashEnv

class SelfPlayCoordinator:
    """
    Manages multiple PixelClashEnv instances for self-play training.
    Handles matchmaking via Clan Friendly Battles (Host/Joiner pair).
    """
    
    def __init__(self, device_ids: List[str]):
        self.device_ids = device_ids
        print(f"Initializing coordinator with {len(device_ids)} devices...")
        
        # Initialize envs in parallel
        with ThreadPoolExecutor() as executor:
            self.envs = list(executor.map(PixelClashEnv, device_ids))
            
        self.num_envs = len(self.envs)
        self.observations = [None] * self.num_envs
        self.dones = [False] * self.num_envs
        self.rewards = [0.0] * self.num_envs
        self.state_history = ["unknown"] * self.num_envs
        
        # Define roles
        # Pair 1: Env 0 (Host) vs Env 1 (Join)
        # Pair 2: Env 2 (Host) vs Env 3 (Join)
        self.hosts = [i for i in range(self.num_envs) if i % 2 == 0]
        self.joiners = [i for i in range(self.num_envs) if i % 2 == 1]

        # Keep OCR lazy-loaded; most coordination now uses templates/color heuristics.
        
    def reset_pair(self, pair_idx: int):
        """Reset a specific pair (Host, Joiner) without stalling others."""
        host_idx = self.hosts[pair_idx]
        joiner_idx = self.joiners[pair_idx]
        host_env = self.envs[host_idx]
        joiner_env = self.envs[joiner_idx]
        
        print(f"\n[Coordinator] Resetting Pair {pair_idx} ({host_idx} vs {joiner_idx})...")
        
        # 0. Quick Check: Are we already in battle?
        # This prevents unnecessary "Middle Clicks" on startup if the devices are already fighting.
        img_h = host_env.capture_screen()
        img_j = joiner_env.capture_screen()
        s_h = host_env.detect_state_blocking(img_h)
        s_j = joiner_env.detect_state_blocking(img_j)
        print(f"  [Pair {pair_idx}] State: Host={s_h}, Joiner={s_j}")

        if s_h == "battle" and s_j == "battle":
            print(f"  [Pair {pair_idx}] Already in Battle. Resuming Training.")
            self.dones[host_idx] = False
            self.dones[joiner_idx] = False
            return
            
        # 1. Back to Chat Recovery
        # Fills standard UI if one is stuck in Menu or End screen
        for env_label, env in [("Host", host_env), ("Joiner", joiner_env)]:
            for _ in range(8):
                img = env.capture_screen()
                state = env.detect_state_blocking(img)
                if state == "clan_chat": break
                
                print(f"  [{env_label}] Recovery: state={state}, taking corrective tap...")
                if state == "menu": 
                    env.tap(env.TAB_SOCIAL_X, int(env.TAB_SOCIAL_Y_RATIO * env._actual_ui_h))
                elif state == "end": 
                    env.tap(env.BTN_OK_X, int(env.BTN_OK_Y_RATIO * env._actual_ui_h))
                else:
                    # OCR Fallback for sticky end screens
                    if env.find_text_and_tap('OK'): pass
                    elif env_label == "Host":
                        # Only the host should ever tap a "Battle" button during recovery.
                        env.find_text_and_tap('Battle')
                time.sleep(1.5)

        # 2. Host Launch
        host_env.find_text_and_tap('Friendly', color='purple')
        time.sleep(1.5)
        host_env.find_text_and_tap('1v1')
        time.sleep(3.0)
        
        # 3. Joiner Accept (Search for YELLOW 'Friendly Battle' button)
        print(f"  [Joiner {joiner_idx}] Waiting for join button...")
        joined = False
        for _ in range(15):
            img = joiner_env.capture_screen()
            # Prefer fast color/shape detection; OCR is a fallback.
            if joiner_env.find_join_button_and_tap(img) or joiner_env.find_text_and_tap('Battle', img=img, color='yellow'):
                print(f"  [Joiner {joiner_idx}] Match Joined!")
                joined = True
                break
            time.sleep(1.0)
        
        if not joined:
            print(f"  WARNING: Joiner {joiner_idx} failed to find join button.")
        
        # 4. Wait for start (Blocking here ensures agents don't start 'Done')
        print(f"  Pair {pair_idx} sequence sent. Waiting for Battle screen...")
        success = False
        for attempt in range(4):
            img_h = host_env.capture_screen()
            img_j = joiner_env.capture_screen()
            s_h = host_env.detect_state(img_h)
            s_j = joiner_env.detect_state(img_j)
            if s_h == "battle" or s_j == "battle" or host_env.is_battle(img_h) or joiner_env.is_battle(img_j):
                print(f"  Pair {pair_idx} successfully in Battle!")
                success = True
                break
            print(f"  Pair {pair_idx} battle not detected (host={s_h}, joiner={s_j}), retrying...")
            time.sleep(5.0)
            
        if not success:
            print(f"  WARNING: Pair {pair_idx} reset timed out. Agents might be stuck.")

        self.dones[host_idx] = False
        self.dones[joiner_idx] = False
        self.state_history[host_idx] = "battle" # Assume battle start
        self.state_history[joiner_idx] = "battle"

    def reset(self) -> np.ndarray:
        """Initial full sync reset."""
        print("Coordinating Initial Sync...")
        # (Same as previous reset lock-step logic but sets readiness)
        # For brevity, calling reset_pair on all pairs sequentially
        for i in range(len(self.hosts)):
            self.reset_pair(i)
        
        # Return first observations
        with ThreadPoolExecutor() as executor:
            imgs = list(executor.map(lambda e: e.capture_screen(), self.envs))
        return np.stack([env.get_observation(imgs[i]) for i, env in enumerate(self.envs)])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step all environments. If both agents in a pair are DONE, auto-reset them.
        """
        # Execute actions
        with ThreadPoolExecutor() as executor:
            def act_single(args):
                i, action = args
                if self.dones[i]: return
                env = self.envs[i]
                env.step_action(action, env.get_elixir(env._last_img))
            executor.map(act_single, enumerate(actions))
            
        time.sleep(1.0)
        
        # Capture
        with ThreadPoolExecutor() as executor:
            imgs = list(executor.map(lambda e: e.capture_screen(), self.envs))
            
        obs_list, rew_list, done_list = [], [], []
        
        for i, env in enumerate(self.envs):
            if self.dones[i]:
                # Preserve last observation, zero reward
                obs_list.append(env.get_observation(env._last_img))
                rew_list.append(0.0)
                done_list.append(True)
                continue
                
            state = env.detect_state_blocking(imgs[i])
            is_done = (state != "battle")
            self.dones[i] = is_done
            
            # Proactive: Clear end screens immediately
            if is_done and state == "end":
                print(f"  [{env.device_id}] Match ended. Clicking OK...")
                env.tap(env.BTN_OK_X, int(env.BTN_OK_Y_RATIO * env._actual_ui_h))
            
            rew = env.estimate_reward(imgs[i])
            obs = env.get_observation(imgs[i])
            
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(is_done)
            env._last_img = imgs[i]
        
        # AUTO-RESET: If both agents in a pair are done, reset them immediately
        for pair_idx in range(len(self.hosts)):
            host_idx = self.hosts[pair_idx]
            joiner_idx = self.joiners[pair_idx]
            if self.dones[host_idx] and self.dones[joiner_idx]:
                print(f"\n[AUTO-RESET] Pair {pair_idx} complete. Starting new match...")
                self.reset_pair(pair_idx)
            
        return np.stack(obs_list), np.array(rew_list, dtype=np.float32), np.array(done_list, dtype=bool), {}
