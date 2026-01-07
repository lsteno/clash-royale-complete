"""
Android Emulator Environment for Clash Royale on macOS.
Uses scrcpy + mss for screen capture and ADB for action injection.
Now supports KataCR perception integration (state + reward) and
ADB screenshot fallback for robustness.
"""
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import mss
import mss.tools
import numpy as np
from typing import TYPE_CHECKING

from src.specs import OBS_SPEC

if TYPE_CHECKING:  # Keep type hints without importing heavy deps at runtime
    from ..perception.katacr_pipeline import KataCRPerceptionEngine, KataCRVisionConfig


@dataclass(frozen=True)
class TapSpec:
    """Simple struct describing a single tap on the emulator screen."""

    x: int
    y: int
    delay: float = 0.5
    label: str = ""


DEFAULT_START_SEQUENCE: Tuple[TapSpec, ...] = (
    TapSpec(980, 320, 0.5, "menu_hamburger"),
    TapSpec(648, 746, 1.0, "training_camp"),
    TapSpec(730, 1400, 2.0, "start_training"),
)

DEFAULT_POST_SEQUENCE: Tuple[TapSpec, ...] = (
    TapSpec(606, 2024, 3.0, "ok_button"),
)


@dataclass
class MatchNavigationConfig:
    """Hard-coded tap sequences to cycle through Training Camp matches."""

    post_match_wait: float = 3.0
    pre_start_wait: float = 1.0
    post_start_wait: float = 2.5
    log_taps: bool = True
    start_sequence: Tuple[TapSpec, ...] = DEFAULT_START_SEQUENCE
    post_match_sequence: Tuple[TapSpec, ...] = DEFAULT_POST_SEQUENCE


class MatchNavigator:
    """Utility class that replays canned tap sequences via ADB."""

    def __init__(self, adb: "ADBController", cfg: MatchNavigationConfig):
        self.adb = adb
        self.cfg = cfg

    def _run_sequence(self, sequence: Sequence[TapSpec]):
        for tap in sequence:
            if self.cfg.log_taps and tap.label:
                print(f"[Navigator] tapping {tap.label} at ({tap.x}, {tap.y})")
            self.adb.tap(tap.x, tap.y)
            if tap.delay > 0:
                time.sleep(tap.delay)

    def dismiss_post_match(self):
        if self.cfg.post_match_wait > 0:
            time.sleep(self.cfg.post_match_wait)
        self._run_sequence(self.cfg.post_match_sequence)

    def start_training_match(self):
        if self.cfg.pre_start_wait > 0:
            time.sleep(self.cfg.pre_start_wait)
        self._run_sequence(self.cfg.start_sequence)
        if self.cfg.post_start_wait > 0:
            time.sleep(self.cfg.post_start_wait)


@dataclass
class EmulatorConfig:
    """Configuration for Android emulator connection."""

    adb_path: str = "adb"
    device_serial: Optional[str] = None  # None for default device
    screen_width: int = 1080
    screen_height: int = 2400
    capture_region: Optional[dict] = None  # {"top": 0, "left": 0, "width": 1080, "height": 2400}
    scrcpy_window_title: str = "scrcpy"  # Window title to capture from
    canonical_width: int = 576  # KataCR expects portrait 1280x576 (H/W ≈ 2.22)
    canonical_height: int = 1280
    enable_adb_fallback: bool = True
    use_adb_capture_only: bool = False  # Use scrcpy (fast screen capture)
    auto_restart: bool = True  # Navigate back to Training Camp after each match automatically
    navigation: MatchNavigationConfig = field(default_factory=MatchNavigationConfig)
    ui_probe_debug: bool = True  # Log sampled UI button colors periodically
    ui_probe_log_every: float = 1.5  # Minimum seconds between probe logs
    ui_probe_save_frames: bool = False  # Save frames when probes fail to match
    ui_probe_dir: Path = Path("logs/ui_probe")


class ADBController:
    """
    Android Debug Bridge controller for action injection.
    Sends tap and swipe commands to the emulator.
    """
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self._check_adb()
    
    def _check_adb(self):
        """Verify ADB is available and device is connected."""
        try:
            result = subprocess.run(
                [self.config.adb_path, "devices"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            devices = [l for l in lines[1:] if l.strip() and 'device' in l]
            if not devices:
                raise RuntimeError("No Android devices/emulators connected. Start Android Studio Emulator first.")
            print(f"Connected devices: {devices}")
        except FileNotFoundError:
            raise RuntimeError(f"ADB not found at {self.config.adb_path}. Install Android SDK or set correct path.")
    
    def _adb_cmd(self, *args) -> str:
        """Execute ADB command."""
        cmd = [self.config.adb_path]
        if self.config.device_serial:
            cmd.extend(["-s", self.config.device_serial])
        cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout
    
    def tap(self, x: int, y: int, duration_ms: int = 50):
        """
        Tap at screen coordinates.
        
        Args:
            x: X coordinate (0 = left)
            y: Y coordinate (0 = top)
            duration_ms: Tap duration in milliseconds
        """
        self._adb_cmd("shell", "input", "tap", str(x), str(y))
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """
        Swipe from (x1, y1) to (x2, y2).
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates  
            duration_ms: Swipe duration
        """
        self._adb_cmd("shell", "input", "swipe", 
                      str(x1), str(y1), str(x2), str(y2), str(duration_ms))
    
    def long_press(self, x: int, y: int, duration_ms: int = 500):
        """Long press at coordinates."""
        self.swipe(x, y, x, y, duration_ms)
    
    def key_event(self, keycode: int):
        """Send key event (e.g., KEYCODE_BACK = 4)."""
        self._adb_cmd("shell", "input", "keyevent", str(keycode))


class ADBScreenshotter:
    """Fallback frame capture using `adb exec-out screencap -p`."""

    def __init__(self, config: EmulatorConfig):
        self.config = config

    def capture(self) -> np.ndarray:
        result = subprocess.run(
            [self.config.adb_path, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode != 0:
            raise RuntimeError(f"adb screencap failed: {result.stderr}")
        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode adb screenshot")
        return img


class ScreenCapture:
    """
    High-performance screen capture using mss.
    Captures the scrcpy mirror window.
    """
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.sct = None if config.use_adb_capture_only else mss.mss()
        self._monitor = None
        if config.enable_adb_fallback or config.use_adb_capture_only:
            self._adb_fallback = ADBScreenshotter(config)
        else:
            self._adb_fallback = None
    
    def _find_scrcpy_window(self) -> Optional[dict]:
        """
        Find the scrcpy window region.
        On macOS, auto-detect the window position using AppleScript.
        """
        # Use explicitly configured region if provided
        if self.config.capture_region:
            return self.config.capture_region
        
        # Try to auto-detect the scrcpy window on macOS
        try:
            from src.utils.window_finder import find_window_bounds
            bounds = find_window_bounds(self.config.scrcpy_window_title)
            if bounds:
                print(f"[ScreenCapture] Auto-detected scrcpy window: {bounds}")
                return bounds
        except Exception as e:
            print(f"[ScreenCapture] Window auto-detection failed: {e}")
        
        # Fallback to primary monitor
        print("[ScreenCapture] Using full primary monitor (set capture_region for better performance)")
        return self.sct.monitors[1]  # monitors[0] is "all monitors"
    
    def capture(self) -> np.ndarray:
        """
        Capture current frame from emulator screen.
        
        Returns:
            np.ndarray: BGR image array (H, W, 3)
        """
        frame = None
        if self.config.use_adb_capture_only:
            if self._adb_fallback is None:
                raise RuntimeError("ADB capture requested but fallback is unavailable.")
            frame = self._adb_fallback.capture()
        else:
            try:
                if self._monitor is None:
                    self._monitor = self._find_scrcpy_window()
                screenshot = self.sct.grab(self._monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception:
                if self._adb_fallback is None:
                    raise
                frame = self._adb_fallback.capture()

        return self._normalize_frame(frame)
    
    def capture_rgb(self) -> np.ndarray:
        """Capture frame in RGB format."""
        frame = self.capture()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        target_size = (self.config.canonical_width, self.config.canonical_height)
        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        return frame


class ClashRoyaleEmulatorEnv:
    """
    Gymnasium-compatible environment for Clash Royale via Android emulator.
    """
    
    # Arena grid dimensions (match OBS_SPEC: width=18, height=32)
    GRID_WIDTH = OBS_SPEC.width
    GRID_HEIGHT = OBS_SPEC.height
    
    # Action space: (card_index, grid_x, grid_y)
    # card_index: 0 = no action, 1-4 = select card
    NUM_CARDS = 4
    
    def __init__(self, config: Optional[EmulatorConfig] = None):
        self.config = config or EmulatorConfig()
        self.adb = ADBController(self.config)
        self.screen = ScreenCapture(self.config)
        self.navigator = (
            MatchNavigator(self.adb, self.config.navigation)
            if self.config.auto_restart
            else None
        )
        self._last_ui_probe_log = 0.0
        self._ok_hit_streak = 0  # consecutive frames meeting OK-button color
        
        # Screen regions (normalized 0-1)
        self.arena_region = {
            "top": 0.15,      # Start after top bar
            "bottom": 0.75,   # End before cards
            "left": 0.0,
            "right": 1.0
        }
        self.cards_region = {
            "top": 0.82,
            "bottom": 0.95,
            "left": 0.15,
            "right": 0.85
        }
        
    def get_observation(self) -> np.ndarray:
        """Get current screen frame."""
        return self.screen.capture_rgb()

    def get_observation_bgr(self) -> np.ndarray:
        """Get current screen frame in BGR (for KataCR pipeline)."""
        return self.screen.capture()

    def _screen_to_frame_coords(self, x: int, y: int, frame: np.ndarray) -> Tuple[int, int]:
        """Scale raw screen coords (e.g., 1080x2400) into the normalized frame size."""
        fx = int(round(x * frame.shape[1] / self.config.screen_width))
        fy = int(round(y * frame.shape[0] / self.config.screen_height))
        fx = int(np.clip(fx, 0, frame.shape[1] - 1))
        fy = int(np.clip(fy, 0, frame.shape[0] - 1))
        return fx, fy

    @staticmethod
    def _matches_color(pixel: np.ndarray, target_bgr: Tuple[int, int, int], tol: int = 30) -> bool:
        """Check if a BGR pixel is within tolerance of a target color."""
        return bool(np.all(np.abs(pixel.astype(int) - np.array(target_bgr)) <= tol))

    @staticmethod
    def _color_distance(pixel: np.ndarray, target_bgr: Tuple[int, int, int]) -> int:
        """Return max-channel absolute distance for quick diagnostics."""
        return int(np.max(np.abs(pixel.astype(int) - np.array(target_bgr))))

    @staticmethod
    def _sample_patch(frame: np.ndarray, center: Tuple[int, int], radius: int = 2) -> np.ndarray:
        """Average BGR color in a (2r+1)x(2r+1) patch around center."""
        cx, cy = center
        x0, x1 = max(0, cx - radius), min(frame.shape[1], cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(frame.shape[0], cy + radius + 1)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return frame[cy, cx]
        return patch.mean(axis=(0, 1))

    def is_match_over(self) -> bool:
        """Detect end-of-match UI by sampling the OK button color and debouncing across frames."""
        frame = self.get_observation_bgr()  # BGR, already normalized to canonical size

        # Screen-space coordinates provided by user (for 1080x2400 reference)
        ok_screen = (613, 2021)      # expected color #44beff (BGR: 255,190,68)

        ok_px = self._screen_to_frame_coords(*ok_screen, frame)

        ok_pixel = frame[ok_px[1], ok_px[0]]

        # Use a small patch average to avoid resize artifacts
        ok_mean = self._sample_patch(frame, ok_px, radius=2)

        ok_color = (255, 187, 104)
        ok_hit = self._matches_color(ok_mean, ok_color, tol=40)

        # One-shot detection: end match as soon as the probe hits.
        self._ok_hit_streak = 1 if ok_hit else 0
        match_over = ok_hit

        now = time.time()
        if self.config.ui_probe_debug and (now - self._last_ui_probe_log) >= self.config.ui_probe_log_every:
            self._last_ui_probe_log = now
            ok_dist = self._color_distance(ok_mean, ok_color)
            print(
                f"[UIProbe] ok@{ok_px} bgr={ok_pixel.tolist()} mean={ok_mean.round(1).tolist()} dist={ok_dist} hit={ok_hit} streak={self._ok_hit_streak}"
            )

            if self.config.ui_probe_save_frames:
                # Save an annotated frame showing where the probe sampled.
                annotated = frame.copy()
                cv2.circle(annotated, ok_px, 6, (0, 0, 255), thickness=2)
                cv2.putText(
                    annotated,
                    f"ok@{ok_px} dist={ok_dist} hit={ok_hit} streak={self._ok_hit_streak}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                probe_dir = Path(self.config.ui_probe_dir)
                probe_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(now * 1000)
                out_path = probe_dir / f"ui_probe_{timestamp}.png"
                cv2.imwrite(str(out_path), annotated)
                print(f"[UIProbe] saved annotated frame to {out_path}")

        return match_over
    
    def get_arena_frame(self) -> np.ndarray:
        """Get only the arena portion of the screen."""
        frame = self.get_observation()
        h, w = frame.shape[:2]
        
        top = int(h * self.arena_region["top"])
        bottom = int(h * self.arena_region["bottom"])
        left = int(w * self.arena_region["left"])
        right = int(w * self.arena_region["right"])
        
        return frame[top:bottom, left:right]
    
    def _grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert arena grid coordinates to screen pixel coordinates."""
        h, w = self.config.screen_height, self.config.screen_width
        
        arena_top = int(h * self.arena_region["top"])
        arena_bottom = int(h * self.arena_region["bottom"])
        arena_left = int(w * self.arena_region["left"])
        arena_right = int(w * self.arena_region["right"])
        
        arena_h = arena_bottom - arena_top
        arena_w = arena_right - arena_left
        
        # Map grid to pixel
        # Use (dim - 1) to keep taps inside the arena even at the max index
        pixel_x = arena_left + int((grid_x / max(1, self.GRID_WIDTH - 1)) * arena_w)
        pixel_y = arena_top + int((grid_y / max(1, self.GRID_HEIGHT - 1)) * arena_h)
        
        return pixel_x, pixel_y
    
    def _card_to_screen(self, card_index: int) -> Tuple[int, int]:
        """Get screen coordinates for card selection (1-4)."""
        h, w = self.config.screen_height, self.config.screen_width
        
        cards_top = int(h * self.cards_region["top"])
        cards_bottom = int(h * self.cards_region["bottom"])
        cards_left = int(w * self.cards_region["left"])
        cards_right = int(w * self.cards_region["right"])
        
        cards_w = cards_right - cards_left
        card_width = cards_w // 4
        
        pixel_x = cards_left + int((card_index - 0.5) * card_width)
        pixel_y = (cards_top + cards_bottom) // 2
        
        return pixel_x, pixel_y
    
    def step(self, action: Tuple[int, int, int]) -> np.ndarray:
        """
        Execute action in environment.
        
        Args:
            action: (card_index, grid_x, grid_y)
                - card_index: 0 = no action, 1-4 = deploy card
                - grid_x: 0-31 arena x position
                - grid_y: 0-17 arena y position
        
        Returns:
            observation: Current screen frame after action
        """
        card_idx, grid_x, grid_y = action
        
        if card_idx > 0:
            # First tap: select card
            card_x, card_y = self._card_to_screen(card_idx)
            self.adb.tap(card_x, card_y)
            time.sleep(0.05)
            
            # Second tap: deploy at position
            deploy_x, deploy_y = self._grid_to_screen(grid_x, grid_y)
            self.adb.tap(deploy_x, deploy_y)
        
        # Small delay for game to process
        time.sleep(0.1)
        
        return self.get_observation()
    
    def reset(self):
        """Reset environment (start new battle if needed)."""
        # This would need game-specific logic
        pass


class FPSLogger:
    """Simple FPS tracker to monitor end-to-end latency."""

    def __init__(self, window: int = 30):
        self.window = window
        self.times = deque(maxlen=window)

    def tick(self) -> Optional[float]:
        now = time.time()
        self.times.append(now)
        if len(self.times) < 2:
            return None
        span = self.times[-1] - self.times[0]
        if span <= 0:
            return None
        return (len(self.times) - 1) / span


class ClashRoyaleKataCREnv(ClashRoyaleEmulatorEnv):
    """
    Emulator environment wired to KataCR perception (state + reward).

    Usage:
    env = ClashRoyaleKataCREnv()
    result, fps, frame = env.capture_state()
      # or inside a loop:
      env.step(action)
    result, fps, frame = env.capture_state()
    """

    def __init__(self, config: Optional[EmulatorConfig] = None, katacr_cfg: Optional["KataCRVisionConfig"] = None):
        super().__init__(config)
        # Lazy import to avoid pulling heavy KataCR/JAX deps on lightweight clients
        from ..perception.katacr_pipeline import KataCRPerceptionEngine

        self.katacr = KataCRPerceptionEngine(katacr_cfg)
        self.fps_logger = FPSLogger()
        self._match_active = False
        self._pending_post_match_cleanup = False

    def reset(self):
        self.katacr.reset()
        if not self.config.auto_restart or self.navigator is None:
            return
        if self._pending_post_match_cleanup:
            self.navigator.dismiss_post_match()
            self._pending_post_match_cleanup = False
        if self._match_active:
            # Already in the middle of a match; nothing else to do.
            return
        self.navigator.start_training_match()
        self._match_active = True

    def mark_match_finished(self):
        if not self.config.auto_restart or self.navigator is None:
            return
        self._match_active = False
        self._pending_post_match_cleanup = True

    def capture_state(self, deploy_cards: Optional[Iterable[str]] = None):
        """Capture a frame, run KataCR perception, and return state, fps, frame."""
        frame_bgr = self.get_observation_bgr()
        result = self.katacr.process(frame_bgr, deploy_cards=deploy_cards)
        fps = self.fps_logger.tick()
        return result, fps, frame_bgr


if __name__ == "__main__":
    # Test the emulator connection
    print("Testing Android Emulator Environment...")
    
    config = EmulatorConfig()
    
    try:
        env = ClashRoyaleEmulatorEnv(config)
        print("✓ ADB connection successful")
        
        frame = env.get_observation()
        print(f"✓ Screen capture working: {frame.shape}")
        
        # Save test frame
        cv2.imwrite("/tmp/test_capture.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("✓ Saved test frame to /tmp/test_capture.png")
        
    except Exception as e:
        print(f"✗ Error: {e}")
