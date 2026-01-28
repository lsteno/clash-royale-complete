import time
import subprocess
from pathlib import Path
import cv2
import numpy as np

# ============== Screen Coordinates (1080x2400) ==============
# ============== Screen Ratios (Based on 1080 Width) ==============
CARD_Y_RATIO = 0.89  
CARD_X = [350, 550, 750, 950]

DEPLOY_POSITIONS = [
    # Bridges (aggressive)
    (250, 0.48, "left_bridge"),
    (550, 0.48, "center_bridge"),
    (850, 0.48, "right_bridge"),
    # Mid positions
    (100, 0.62, "left_mid"),
    (550, 0.60, "center_mid"),
    (950, 0.62, "right_mid"),
    # Back positions (defensive)
    (100, 0.71, "left_back"),
    (550, 0.73, "center_back"),
    (950, 0.71, "right_back"),
]

class PixelClashEnv:
    """
    Control a single Clash Royale instance via ADB.
    Returns pixel observations (128x256).
    """
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.adb_prefix = ['adb', '-s', device_id]
        self.state = "unknown"
        self._last_battle_ts = 0.0
        self._templates_loaded = False
        self._templates = {}
        self._last_img = None
        self._native_wh = None 
        self._scale = 1.0
        self._actual_ui_h = 2400 
        # State-based HP for rewards [EnemyP1, EnemyP2, EnemyKing, OurP1, OurP2, OurKing]
        self.last_hp = [1.0] * 6

    def _adb_command(self, cmd: list, timeout: float = 2.0) -> bytes:
        try:
            result = subprocess.run(
                self.adb_prefix + cmd,
                capture_output=True,
                timeout=timeout
            )
            return result.stdout if result.returncode == 0 else None
        except:
            return None

    def tap(self, x: int, y: int):
        """Logic (1080x2400) -> Native."""
        sx = int(x / self._scale)
        sy = int(y / self._scale)
        self._adb_command(['shell', 'input', 'tap', str(sx), str(sy)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 150):
        """Logic (1080x2400) -> Native."""
        sx1, sy1 = int(x1 / self._scale), int(y1 / self._scale)
        sx2, sy2 = int(x2 / self._scale), int(y2 / self._scale)
        self._adb_command(['shell', 'input', 'swipe', 
                          str(sx1), str(sy1), str(sx2), str(sy2), str(duration)])

    def capture_screen(self) -> np.ndarray:
        """Capture screen via ADB with Aspect-Ratio Preserving Scaling (Fit-to-Width)."""
        data = self._adb_command(['exec-out', 'screencap', '-p'], timeout=1.5)
        if data is None: return None
        try:
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: return None
            
            h_h, w_w = img.shape[:2]
            if self._native_wh is None:
                self._native_wh = (w_w, h_h)
                self._scale = 1080 / w_w
                self._actual_ui_h = int(h_h * self._scale)
                print(f"[{self.device_id}] Native: {w_w}x{h_h}, Scale: {self._scale:.2f}, Logic Active H: {self._actual_ui_h}")
            
            # Scale to width 1080
            resized = cv2.resize(img, (1080, self._actual_ui_h))
            
            # Pad to 2400
            canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
            canvas[:self._actual_ui_h, :] = resized
            
            self._last_img = canvas
            return canvas
        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def get_observation(self, img: np.ndarray = None) -> np.ndarray:
        """
        Process image for Dreamer (C, H, W).
        Battlefield only: Excludes card deck at the bottom.
        """
        if img is None:
            img = self.capture_screen()
        if img is None:
            return np.zeros((3, 256, 128), dtype=np.float32)
            
        # Extract arena region (Dynamic based on Active H)
        # Ends above the card UI (approx 82% of screen)
        ay1 = int(self._actual_ui_h * 0.05)
        ay2 = int(self._actual_ui_h * 0.82)
        arena = img[ay1:ay2, 22:1058]
        
        arena = cv2.resize(arena, (128, 256))
        arena = cv2.cvtColor(arena, cv2.COLOR_BGR2RGB)
        arena = arena.astype(np.float32) / 255.0
        return arena.transpose(2, 0, 1)

    # Clan UI
    # UI Ratios
    TAB_SOCIAL_X = 950
    TAB_SOCIAL_Y_RATIO = 0.92
    
    BTN_FRIENDLY_X = 800
    BTN_FRIENDLY_Y_RATIO = 0.08
    
    BTN_CONFIRM_1v1_X = 750
    BTN_CONFIRM_1v1_Y_RATIO = 0.66
    
    BTN_ACCEPT_X = 800
    BTN_ACCEPT_Y_RATIO = 0.58
    
    BTN_OK_X = 540
    BTN_OK_Y_RATIO = 0.91

    def _state_rois(self):
        """Return ROIs for state templates in logic coordinates."""
        return {
            # Elixir is a thin bar very close to the bottom; keep this ROI tight.
            "elixir": (280, int(self._actual_ui_h * 0.965), 1050, int(self._actual_ui_h * 0.985)),
            "ok_button": (400, int(self._actual_ui_h * 0.85), 700, int(self._actual_ui_h * 0.92)),
            "friendly_button": (320, int(self._actual_ui_h * 0.78), 760, int(self._actual_ui_h * 0.90)),
            "cancel_button": (820, int(self._actual_ui_h * 0.02), 1060, int(self._actual_ui_h * 0.12)),
            "menu_battle_button": (360, int(self._actual_ui_h * 0.72), 720, int(self._actual_ui_h * 0.86)),
        }

    def _safe_roi_crop(self, img: np.ndarray, roi):
        x1, y1, x2, y2 = roi
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _battle_rois(self):
        """
        ROIs for battle detection.
        Keep them simple and robust across minor UI shifts.
        """
        h = self._actual_ui_h
        return {
            # Lower band where elixir bar lives (wider than template ROI).
            "elixir_band": (220, int(h * 0.94), 1060, int(h * 0.992)),
            # Mid-field where arena grass dominates during battle.
            "arena_mid": (80, int(h * 0.22), 1000, int(h * 0.70)),
            # Bottom-right where join "Friendly Battle" button appears in clan chat.
            "join_button": (560, int(h * 0.72), 1068, int(h * 0.93)),
        }

    def _load_templates(self):
        if self._templates_loaded:
            return
        template_dir = Path(__file__).resolve().parent / "templates"
        templates = {
            "ok_button": [template_dir / "ok_button.png"],
            "cancel_button": [template_dir / "cancel_button.png"],
            "friendly_button": [template_dir / "friendly_button.png"],
            "menu_battle_button": [template_dir / "menu_battle_button.png"],
            "elixir": [template_dir / "elixir.png", template_dir / "elixir_bar.png"],
        }
        for name, paths in templates.items():
            for path in paths:
                if path.exists():
                    img = cv2.imread(str(path))
                    if img is not None:
                        self._templates[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        break
        self._templates_loaded = True

    def _match_template(self, img: np.ndarray, roi, template_name: str) -> float:
        self._load_templates()
        tpl = self._templates.get(template_name)
        if tpl is None:
            return 0.0
        crop = self._safe_roi_crop(img, roi)
        if crop is None:
            return 0.0
        crop_g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop_g.shape[0] < tpl.shape[0] or crop_g.shape[1] < tpl.shape[1]:
            return 0.0
        res = cv2.matchTemplate(crop_g, tpl, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def _color_ratio(self, img: np.ndarray, roi, h_min, h_max, s_min, invert_h=False) -> float:
        crop = self._safe_roi_crop(img, roi)
        if crop is None:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if invert_h:
            mask = ((hsv[:, :, 0] < h_min) | (hsv[:, :, 0] > h_max)) & (hsv[:, :, 1] > s_min)
        else:
            mask = (hsv[:, :, 0] > h_min) & (hsv[:, :, 0] < h_max) & (hsv[:, :, 1] > s_min)
        return float(mask.mean())

    def _pink_column_coverage(self, img: np.ndarray, roi) -> float:
        """
        Detect elixir-like magenta in a bottom band and return horizontal coverage [0..1].
        Uses column coverage to be robust against small overlays.
        """
        crop = self._safe_roi_crop(img, roi)
        if crop is None:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Magenta/pink (OpenCV hue 0..179): ~130..175 with decent saturation/value.
        mask = (hsv[:, :, 0] > 130) & (hsv[:, :, 0] < 175) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)
        # A column is considered "active" if enough pixels in that column match.
        col_active = (mask.mean(axis=0) > 0.12)
        return float(col_active.mean())

    def _green_ratio(self, img: np.ndarray, roi) -> float:
        """Arena grass proxy (helps disambiguate chat/menu from battle)."""
        crop = self._safe_roi_crop(img, roi)
        if crop is None:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Arena grass skews yellow-green on many devices; keep range broad.
        mask = (hsv[:, :, 0] > 18) & (hsv[:, :, 0] < 60) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
        return float(mask.mean())

    def is_battle(self, img: np.ndarray) -> bool:
        """
        Simple and robust battle detection:
        - Elixir magenta band near the bottom.

        This intentionally avoids OCR/template dependencies and is tuned to minimize
        false negatives during loading/overlays.
        """
        if img is None or img.size == 0:
            return False
        rois = self._battle_rois()
        pink_cov = self._pink_column_coverage(img, rois["elixir_band"])
        green = self._green_ratio(img, rois["arena_mid"])

        # Strong signal: a wide magenta band looks like the elixir UI (battle-only).
        if pink_cov > 0.60:
            return True

        # Typical battle: mostly green arena + at least some magenta UI presence.
        if green > 0.22 and pink_cov > 0.04:
            return True

        # Fallback for edge cases (e.g., elixir visually depleted/covered).
        return green > 0.55

    def detect_state(self, img) -> str:
        """
        Detect game state: 'battle', 'end', 'menu', 'clan_chat', or 'unknown'.
        
        Rules (template-first with color fallback):
        1. End if OK button template/color matches.
        2. Clan chat if Friendly Battle or Cancel template/color matches.
        3. Menu if yellow Battle button template/color matches.
        4. Battle if elixir bar template/color matches.
        """
        if img is None or img.size == 0:
            return "unknown"

        rois = self._state_rois()
        ok_score = self._match_template(img, rois["ok_button"], "ok_button")
        # Keep end detection strict: color-only checks are prone to false positives on battle UI.
        if ok_score >= 0.75:
            self.state = "end"
            return "end"

        if self.is_battle(img):
            self._last_battle_ts = time.time()
            self.state = "battle"
            return "battle"

        friendly_score = self._match_template(img, rois["friendly_button"], "friendly_button")
        cancel_score = self._match_template(img, rois["cancel_button"], "cancel_button")
        friendly_color = self._color_ratio(img, rois["friendly_button"], 125, 170, 80)
        cancel_color = self._color_ratio(img, rois["cancel_button"], 10, 170, 100, invert_h=True)
        if max(friendly_score, cancel_score) >= 0.72 or friendly_color > 0.12 or cancel_color > 0.08:
            self.state = "clan_chat"
            return "clan_chat"

        menu_score = self._match_template(img, rois["menu_battle_button"], "menu_battle_button")
        menu_color = self._color_ratio(img, rois["menu_battle_button"], 20, 45, 100)
        if menu_score >= 0.72 or menu_color > 0.08:
            self.state = "menu"
            return "menu"

        # Short battle grace period to ride out transient misses.
        # Require at least some magenta presence in the elixir band to avoid
        # misclassifying clan/menu screens as battle.
        if time.time() - self._last_battle_ts < 10.0:
            rois_b = self._battle_rois()
            if self._pink_column_coverage(img, rois_b["elixir_band"]) > 0.06:
                self.state = "battle"
                return "battle"

        self.state = "unknown"
        return "unknown"

    def detect_state_with_retry(self, img: np.ndarray = None, retries: int = 4, delay_s: float = 5.0) -> str:
        """Retry state detection to ride out loading/transition screens."""
        if img is None:
            img = self.capture_screen()
        state = self.detect_state(img)
        if state != "unknown":
            return state
        for _ in range(retries):
            time.sleep(delay_s)
            img = self.capture_screen()
            state = self.detect_state(img)
            if state != "unknown":
                return state
        return state

    def detect_state_blocking(self, img: np.ndarray = None, delay_s: float = 2.0, timeout_s: float = 25.0) -> str:
        """Block until a non-unknown state is detected (with timeout)."""
        if img is None:
            img = self.capture_screen()
        state = self.detect_state(img)
        start = time.time()
        while state == "unknown" and (time.time() - start) < timeout_s:
            time.sleep(delay_s)
            img = self.capture_screen()
            state = self.detect_state(img)
        if state == "unknown":
            # Prefer returning the last known state to avoid deadlocks in coordination loops.
            return getattr(self, "state", "unknown")
        return state
    
    def _ocr_detect_text(self, img, search_terms: list) -> bool:
        """
        Search for any of the given terms in the image using OCR.
        Uses fuzzy matching to handle 1-2 character OCR errors.
        """
        try:
            reader = self._get_reader()
            result = reader.readtext(img, paragraph=False)
            
            # Extract all detected text (case-insensitive)
            detected_texts = [text.lower() for _, text, _ in result]
            
            # Check if any search term appears in detected text
            for term in search_terms:
                term_lower = term.lower()
                for detected in detected_texts:
                    # Fuzzy match: allow term to be substring or very close
                    if term_lower in detected or detected in term_lower:
                        return True
                    # Also check Levenshtein distance for typos
                    if len(term_lower) > 3 and len(detected) > 3:
                        if self._fuzzy_match(term_lower, detected, max_dist=2):
                            return True
            return False
        except Exception:
            return False
    
    def _fuzzy_match(self, s1: str, s2: str, max_dist: int = 2) -> bool:
        """Simple fuzzy string matching using Levenshtein-like distance."""
        if abs(len(s1) - len(s2)) > max_dist:
            return False
        
        # Count character differences (simple approach)
        min_len = min(len(s1), len(s2))
        differences = sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len]))
        differences += abs(len(s1) - len(s2))
        
        return differences <= max_dist

    def estimate_reward(self, curr_img: np.ndarray) -> float:
        """
        Calculate reward based on reduction in health bar pixel counts.
        Uses 6 precision ROIs verified by the user.
        """
        if curr_img is None: return 0.0

        # Final 精确 ROI (Verified with static calibration tool)
        # Sequence: [EnemyP1, EnemyP2, EnemyKing, OurP1, OurP2, OurKing]
        regions = [
            (216, 333, int(self._actual_ui_h * 0.146), int(self._actual_ui_h * 0.155)), # Enemy P1
            (783, 903, int(self._actual_ui_h * 0.148), int(self._actual_ui_h * 0.156)), # Enemy P2
            (478, 648, int(self._actual_ui_h * 0.030), int(self._actual_ui_h * 0.042)), # Enemy King
            (218, 338, int(self._actual_ui_h * 0.619), int(self._actual_ui_h * 0.630)), # Our P1
            (785, 903, int(self._actual_ui_h * 0.620), int(self._actual_ui_h * 0.630)), # Our P2
            (486, 643, int(self._actual_ui_h * 0.748), int(self._actual_ui_h * 0.767)), # Our King
        ]

        def get_bar_fullness(img, reg, is_blue):
            x1, x2, y1, y2 = reg
            area = img[y1:y2, x1:x2]
            if area.size == 0: return 0.0
            hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
            if is_blue:
                mask = (hsv[:,:,0] > 100) & (hsv[:,:,0] < 125) & (hsv[:,:,1] > 100)
            else:
                mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 100)
            
            return mask.mean()

        reward = 0.0
        enemy_reward = 0.0
        our_penalty = 0.0
        # Check first 3 (Enemy - Reward for drop)
        for i in range(3):
            fullness = get_bar_fullness(curr_img, regions[i], is_blue=False)
            
            # Visibility Check (ignore if bar hasn't appeared yet)
            if fullness < 0.02:
                if self.last_hp[i] > 0.1: # Transition from visible to gone! (Destroyed)
                    delta = (self.last_hp[i] * 10.0)
                    reward += delta
                    enemy_reward += delta
                    self.last_hp[i] = 0.0
                continue # Skip rewards while hidden
                
            if fullness < self.last_hp[i] - 0.01: 
                diff = self.last_hp[i] - fullness
                delta = (diff * 10.0)
                reward += delta
                enemy_reward += delta
                self.last_hp[i] = fullness
            elif fullness > self.last_hp[i] + 0.1: # Reset/Appearance
                 self.last_hp[i] = fullness
        
        # Check last 3 (Ours - Penalty for drop)
        for i in range(3, 6):
            fullness = get_bar_fullness(curr_img, regions[i], is_blue=True)
            
            if fullness < 0.02:
                if self.last_hp[i] > 0.1: 
                    delta = (self.last_hp[i] * 10.0)
                    reward -= delta
                    our_penalty += delta
                    self.last_hp[i] = 0.0
                continue

            if fullness < self.last_hp[i] - 0.01:
                diff = self.last_hp[i] - fullness
                delta = (diff * 10.0)
                reward -= delta
                our_penalty += delta
                self.last_hp[i] = fullness
            elif fullness > self.last_hp[i] + 0.1:
                 self.last_hp[i] = fullness

        self.last_reward_breakdown = {
            "enemy": enemy_reward,
            "ours": -our_penalty,
            "total": reward,
        }
        return reward

    def get_elixir(self, img: np.ndarray) -> int:
        """Read elixir from relative bottom region."""
        if img is None: return 0
        rois = self._battle_rois()
        elixir_bar = self._safe_roi_crop(img, rois["elixir_band"])
        if elixir_bar is None:
            return 0

        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 0] > 130) & (hsv[:, :, 0] < 175) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)
        cols = (mask.mean(axis=0) > 0.12)
        if cols.any():
            filled = int(np.where(cols)[0].max())
            return int(min(10, max(0, round((filled / max(1, mask.shape[1] - 1)) * 10))))
        return 0

    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if not hasattr(self, '_reader') or self._reader is None:
            print("Initializing OCR Reader (this may take a moment)...")
            import easyocr
            # EasyOCR's GPU path uses DataLoader pinning which is noisy/unsupported on MPS.
            # Prefer CUDA when available; otherwise use CPU for stability.
            try:
                import torch

                use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            except Exception:
                use_gpu = False
            self._reader = easyocr.Reader(['en'], gpu=use_gpu)
        return self._reader

    def find_join_button_and_tap(self, img: np.ndarray = None) -> bool:
        """Tap the join button in clan chat via color/shape (no OCR)."""
        if img is None:
            img = self.capture_screen()
        if img is None:
            return False
        roi = self._battle_rois()["join_button"]
        crop = self._safe_roi_crop(img, roi)
        if crop is None:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Yellow/Orange-ish button background.
        mask = (hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 45) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)
        mask_u8 = (mask.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        # Pick the largest yellow blob.
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < 800.0:
            return False
        x, y, w, h = cv2.boundingRect(cnt)
        # Tap center in full-image coordinates.
        cx = roi[0] + x + w // 2
        cy = roi[1] + y + h // 2
        self.tap(int(cx), int(cy))
        return True

    def find_text_and_tap(self, target_text: str, img: np.ndarray = None, color: str = None) -> bool:
        """
        Locate and tap UI element. 
        Special Logic for '1v1': Picks topmost 'Battle' button in top-half.
        Standard Logic: Fuzzy string + optional color.
        """
        if img is None:
            img = self.capture_screen()
        if img is None: return False
            
        reader = self._get_reader()
        results = reader.readtext(img)
        hsv = None
        if color: hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        target_lower = target_text.lower()
        import difflib
        
        # 1. SPECIAL CASE: 1v1 (Anchor-based)
        if target_lower == '1v1':
            anchor_y = -1
            for (bbox, text, prob) in results:
                if "basic" in text.lower() and prob > 0.8:
                    anchor_y = int((bbox[0][1] + bbox[2][1]) / 2)
                    # print(f"  Found Anchor 'BASIC' at y={anchor_y}")
                    break
            
            if anchor_y != -1:
                # Look for 'Battle' below BASIC
                for (bbox, text, prob) in results:
                    t_low = text.lower()
                    cy = int((bbox[0][1] + bbox[2][1]) / 2)
                    # Search range: ratio height offset
                    min_y = anchor_y + int(self._actual_ui_h * 0.02)
                    max_y = anchor_y + int(self._actual_ui_h * 0.12)
                    if ("battle" in t_low or "iv" in t_low) and (min_y < cy < max_y):
                        print(f"  [Anchor 1v1] Selected '{text}' relative to BASIC anchor.")
                        (tl, tr, br, bl) = bbox
                        self.tap(int((tl[0] + br[0]) / 2), cy)
                        return True

            # If anchor failed, fallback to specialized 1v1 fuzzy list
            # Tightened: length check and tighter Y range to avoid clicking chat log bubbles
            v1_aliases = ['1v1', 'lvl', 'ivi', '1vi', 'lvi', 'ivl', '1vl', 'iv1', 'iv]', 'basic']
            for (bbox, text, prob) in results:
                t_low = text.lower().strip()
                cy = int((bbox[0][1] + bbox[2][1]) / 2)
                
                # Rule 1: Text must be very short (button label only)
                # Rule 2: Must match an alias
                # Rule 3: Vertical range must be in the top-middle where the pop-up is (20% to 35% height)
                if len(t_low) <= 6 and any(al == t_low for al in v1_aliases):
                    if (self._actual_ui_h * 0.20) < cy < (self._actual_ui_h * 0.45):
                        (tl, tr, br, bl) = bbox
                        cx = int((tl[0] + br[0]) / 2)
                        print(f"  [Fuzzy 1v1 Fallback] Selected '{text}' at ({cx}, {cy})")
                        self.tap(cx, cy)
                        return True
            return False

        # 2. STANDARD CASE: Fuzzy + Color
        best_bbox = None
        highest_score = 0.0
        
        for (bbox, text, prob) in results:
            t_low = text.lower()
            
            # Simple substring or fuzzy
            if target_lower in t_low:
                similarity = 1.0
            else:
                similarity = difflib.SequenceMatcher(None, target_lower, t_low).ratio()
            
            if similarity > 0.7:
                # Color Filter
                if color:
                    # Expand ROI by 10% to ensure we catch background color around the text
                    h_h, w_w = hsv.shape[:2]
                    y1, y2 = max(0, int(bbox[0][1] - 5)), min(h_h, int(bbox[2][1] + 5))
                    x1, x2 = max(0, int(bbox[0][0] - 5)), min(w_w, int(bbox[1][0] + 5))
                    
                    if y2 > y1 and x2 > x1:
                        roi = hsv[y1:y2, x1:x2]
                        # Use 90th percentile to ignore black text and catch the bright button color
                        pixel_h = np.percentile(roi[:,:,0], 90)
                        pixel_s = np.percentile(roi[:,:,1], 90)
                    else:
                        pixel_h, pixel_s = 0, 0
                    
                    match_col = False
                    if color == 'purple':
                        if 115 < pixel_h < 165 and pixel_s > 40: match_col = True
                    elif color == 'yellow':
                        # Yellow/Orange range for buttons
                        if 5 < pixel_h < 50 and pixel_s > 40: match_col = True
                        
                    if not match_col:
                        # print(f"  skip '{text}': wrong color (H={pixel_h:.1f})")
                        continue
                
                score = similarity * prob
                if score > highest_score:
                    highest_score = score
                    best_bbox = bbox
        
        if best_bbox:
            (tl, tr, br, bl) = best_bbox
            cx = int((tl[0] + br[0]) / 2)
            cy = int((tl[1] + br[1]) / 2)
            print(f"OCR match: '{target_text}' at ({cx}, {cy})")
            self.tap(cx, cy)
            return True
            
        return False

    def step_action(self, action_idx: int, elixir: int):
        """
        Execute action using relative coordinates.
        action_idx: 0 (no op) to 36 (4 cards * 9 positions)
        """
        if action_idx == 0:
            return
            
        # Decode action
        card_idx = (action_idx - 1) // len(DEPLOY_POSITIONS) 
        pos_idx = (action_idx - 1) % len(DEPLOY_POSITIONS)
        
        if card_idx >= len(CARD_X):
            print(f"  [{self.device_id}] Invalid card index {card_idx} (max {len(CARD_X)-1})")
            return
            
        # Elixir Safety Check: Prevent empty card plays if below 3 elixir
        if elixir < 3:
            return
            
        # 1. Card selection (Y is relative CARD_Y_RATIO)
        cX = CARD_X[card_idx]
        cY = int(self._actual_ui_h * CARD_Y_RATIO)
 
        # 2. Deployment Swipes
        px_orig, py_ratio, _ = DEPLOY_POSITIONS[pos_idx]
        pX = px_orig
        pY = int(py_ratio * self._actual_ui_h)
        
        print(f"  [{self.device_id}] Play Card {card_idx+1} at ({pX}, {pY})")
        self.swipe(cX, cY, pX, pY)
