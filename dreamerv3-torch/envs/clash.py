
import time
import gymnasium as gym
import numpy as np
import cv2
from clash_env.pixel_env import PixelClashEnv
from clash_env.matchmaker import MatchMaker

class ClashEnv(gym.Env):
    def __init__(self, device_id, size=(256, 128)):
        super().__init__()
        self.device_id = device_id
        self._size = size
        
        # Initialize internal env
        self._env = PixelClashEnv(device_id)
        self._rng = np.random.RandomState()
        self._is_joiner = MatchMaker.get_instance().is_joiner(device_id)
        self.is_joiner = self._is_joiner
        self._joiner_min_delay = 0.5
        self._joiner_max_delay = 3.0
        self._joiner_max_window = 4.0
        self._next_joiner_time = None
        self._last_host_action_ts = None
        
        self._last_img = None
        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        # Dreamer expects Dict with 'image', 'is_first', 'is_terminal'
        return gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (self._size[0], self._size[1], 3), dtype=np.uint8),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        # 0: no-op, 1-4*9: cards (4) * positions (9) = 36 actions
        # Total = 37
        space = gym.spaces.Discrete(37) 
        space.discrete = True
        return space

    def step(self, action):
        # If we are waiting for a partner, don't perform any real action
        mm = MatchMaker.get_instance()
        if self._done:
            if not mm.try_joint_reset(self.device_id):
                # Partner not ready yet. Return dummy obs.
                obs, _, _, info = self._obs(0.0, is_last=False, is_terminal=False)
                return obs, 0.0, False, info
            else:
                # Joint reset just finished!
                img = self._env.capture_screen()
                self._last_img = img
                self._done = False
                obs, _, _, info = self._obs(0.0, is_first=True)
                return obs, 0.0, False, info

        idx = int(action)
        action_override = None

        # 1. Get current elixir
        elixir = self._env.get_elixir(self._last_img)

        mm = MatchMaker.get_instance()
        if self._is_joiner:
            host_ts = mm.get_last_host_action(self.device_id)
            if host_ts is None:
                idx = 0
            else:
                if self._last_host_action_ts != host_ts:
                    self._last_host_action_ts = host_ts
                    self._next_joiner_time = host_ts + self._rng.uniform(
                        self._joiner_min_delay, self._joiner_max_delay
                    )

                now = time.time()
                if (
                    self._next_joiner_time is None
                    or now < self._next_joiner_time
                    or (now - host_ts) > self._joiner_max_window
                    or elixir < 3
                ):
                    idx = 0
                else:
                    idx = self._rng.randint(1, self.action_space.n)
                    self._next_joiner_time = None

            action_override = np.zeros(self.action_space.n, dtype=np.float32)
            action_override[idx] = 1.0
        else:
            if idx != 0 and elixir >= 3:
                mm.mark_host_action(self.device_id)
        
        # 2. Execute action
        self._env.step_action(idx, elixir)
        
        # 3. Capture result
        img = self._env.capture_screen()
        self._last_img = img
        
        # 4. State Detection & Reward
        state = self._env.detect_state(img)
        if state != "battle":
            # Confirm terminal state across multiple frames to avoid false positives.
            states = [state]
            for _ in range(2):
                time.sleep(0.4)
                img2 = self._env.capture_screen()
                if img2 is None:
                    break
                img = img2
                self._last_img = img2
                st2 = self._env.detect_state(img2)
                states.append(st2)
                if st2 == "battle":
                    state = "battle"
                    break
                state = st2
        reward = self._env.estimate_reward(img)
        breakdown = getattr(self._env, "last_reward_breakdown", None)
        if breakdown and breakdown.get("total", 0.0) != 0.0:
            print(
                f"  [{self.device_id}] reward {breakdown['total']:.3f} "
                f"(enemy +{breakdown['enemy']:.3f}, ours {breakdown['ours']:.3f})"
            )
        
        self._done = (state != 'battle')
        
        # Debug: Show what state we detected
        if self._done:
            extra = ""
            try:
                rois = self._env._battle_rois()
                pink = self._env._pink_column_coverage(self._last_img, rois["elixir_band"])
                green = self._env._green_ratio(self._last_img, rois["arena_mid"])
                extra = f" pink_cov={pink:.3f} green={green:.3f}"
            except Exception:
                pass
            if "states" in locals():
                print(f"  [{self.device_id}] Match ended. states={states}{extra}")
            else:
                print(f"  [{self.device_id}] Match ended. State detected: '{state}'{extra}")
        
        # Proactive: If battle just ended, try to stabilize back to chat/menu.
        if self._done:
            state2 = self._env.detect_state_blocking()
            if state2 == "end":
                print(f"  [{self.device_id}] Clicking OK...")
                self._env.tap(self._env.BTN_OK_X, int(self._env.BTN_OK_Y_RATIO * self._env._actual_ui_h))
                time.sleep(1.5)
                state2 = self._env.detect_state_blocking()
                print(f"  [{self.device_id}] After OK, state: '{state2}'")

            if state2 == "menu":
                print(f"  [{self.device_id}] Switching to Clan Chat...")
                self._env.tap(self._env.TAB_SOCIAL_X, int(self._env.TAB_SOCIAL_Y_RATIO * self._env._actual_ui_h))
            elif state2 == "unknown":
                if self._env.find_text_and_tap("OK"):
                    time.sleep(1.5)
                state2 = self._env.detect_state_blocking()
                if state2 == "menu":
                    print(f"  [{self.device_id}] Switching to Clan Chat...")
                    self._env.tap(self._env.TAB_SOCIAL_X, int(self._env.TAB_SOCIAL_Y_RATIO * self._env._actual_ui_h))
        
        obs, _, _, info = self._obs(reward, is_last=self._done, is_terminal=self._done)
        if action_override is not None:
            info["action"] = action_override
        return obs, reward, self._done, info

    def reset(self, seed=None, options=None):
        # Non-blocking reset initiation
        MatchMaker.get_instance().mark_ready(self.device_id)
        self._done = True
        
        # Return initial dummy obs
        obs, _, _, _ = self._obs(0.0, is_first=True)
        return obs

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        if self._last_img is None:
            return {'image': np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8),
                    'is_first': is_first, 'is_terminal': is_terminal}, reward, is_last, {}

        # Arena cropping and resizing
        ay1 = int(self._env._actual_ui_h * 0.05)
        ay2 = int(self._env._actual_ui_h * 0.82)
        arena = self._last_img[ay1:ay2, 22:1058]
        arena = cv2.resize(arena, (self._size[1], self._size[0]))
        arena = cv2.cvtColor(arena, cv2.COLOR_BGR2RGB) # RGB (256, 128, 3)
        
        obs = {
            'image': arena.astype(np.uint8),
            'is_first': is_first,
            'is_terminal': is_terminal,
        }
        return obs, reward, is_last, {}

    def close(self):
        pass
