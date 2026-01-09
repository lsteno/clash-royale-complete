"""
KataCR Perception Adapter for macOS.

Adapts KataCR's perception pipeline to work on macOS:
- Uses ADB screenshots instead of v4l2loopback/scrcpy video stream
- Uses KataCR's YOLOv8 weights with ultralytics (no JAX needed)
- Provides state extraction (elixir, cards, units, time)
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time

# Add KataCR to path
KATACR_PATH = Path(__file__).parents[3] / "KataCR"
sys.path.insert(0, str(KATACR_PATH))

# Try to import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics==8.1.24")
    ULTRALYTICS_AVAILABLE = False

# Load unit labels from KataCR (doesn't need JAX)
try:
    from katacr.constants.label_list import idx2unit, unit2idx
    LABELS_AVAILABLE = True
except ImportError:
    print("Warning: Could not load KataCR labels")
    LABELS_AVAILABLE = False
    idx2unit = {}
    unit2idx = {}


class ADBScreenCapture:
    """Capture screenshots via ADB instead of video stream."""
    
    def __init__(self):
        self.last_img = None
        self.timestamp = 0
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame via ADB screenshot."""
        try:
            # Capture screenshot
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=2
            )
            if result.returncode != 0:
                return False, None
            
            # Decode PNG
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return False, None
            
            self.last_img = img
            self.timestamp = time.time()
            return True, img
            
        except Exception as e:
            print(f"Screenshot error: {e}")
            return False, None
    
    def isOpened(self) -> bool:
        """Check if ADB is available."""
        try:
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                timeout=2
            )
            return b"device" in result.stdout
        except:
            return False


class SimplifiedStateBuilder:
    """
    Simplified state builder that extracts key game state.
    Inspired by KataCR but without heavy ML dependencies.
    """
    
    # Screen regions for 1080x2400 resolution
    REGIONS = {
        # Shifted elixir crops 20px down for better alignment (capture-region 1,38,494,1074)
        'elixir_text': (200, 2320, 400, 2420),  # x1, y1, x2, y2
        'elixir_bar': (200, 2320, 1050, 2420),  # Updated coordinates
        'cards': [(300, 2050, 400, 2250), (500, 2050, 600, 2250),
                  (700, 2050, 800, 2250), (900, 2050, 1000, 2250)],
        'arena': (22, 580, 1058, 1850),
        # Shifted time crop 40px down and 15px right (capture-region 1,38,494,1074)
        'time': (465, 90, 645, 160),
        'our_tower_hp': [(100, 1650, 250, 1750), (830, 1650, 980, 1750), (450, 1750, 630, 1850)],
        'enemy_tower_hp': [(100, 650, 250, 750), (830, 650, 980, 750), (450, 550, 630, 650)],
    }
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.time = 0
        self.elixir = 0
        self.prev_our_hp = [0, 0, 0]  # left princess, right princess, king
        self.prev_enemy_hp = [0, 0, 0]
        
    def extract_elixir(self, img: np.ndarray) -> int:
        """Extract elixir count using pink/purple bar detection."""
        x1, y1, x2, y2 = self.REGIONS['elixir_bar']
        bar = img[y1:y2, x1:x2]
        
        # Convert to HSV and find pink/purple elixir
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        # Pink/purple in HSV: H=140-180 or H=0-15 (wraps), with saturation
        mask = ((hsv[:,:,0] > 140) | (hsv[:,:,0] < 15)) & (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
        
        # Calculate filled percentage based on pink pixel density
        # The bar fills from left to right
        cols = mask.any(axis=0)
        if cols.any():
            # Find rightmost pink column
            filled = np.where(cols)[0].max()
            bar_width = bar.shape[1]
            elixir = int((filled / bar_width) * 10)
            return min(10, max(0, elixir))
        return 0
    
    def extract_arena_features(self, img: np.ndarray) -> np.ndarray:
        """Extract arena region and resize for model input."""
        x1, y1, x2, y2 = self.REGIONS['arena']
        arena = img[y1:y2, x1:x2]
        # Resize to standard size for model
        arena_resized = cv2.resize(arena, (128, 256))
        return arena_resized
    
    def detect_units_simple(self, img: np.ndarray) -> List[Dict]:
        """
        Simple unit detection using color and motion.
        For proper detection, use KataCR's YOLOv8 model.
        """
        x1, y1, x2, y2 = self.REGIONS['arena']
        arena = img[y1:y2, x1:x2]
        
        units = []
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
        
        # Detect red (enemy) units - red health bars
        red_mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 100) & (hsv[:,:,2] > 100)
        
        # Detect blue (friendly) units - blue health bars
        blue_mask = (hsv[:,:,0] > 100) & (hsv[:,:,0] < 130) & (hsv[:,:,1] > 100) & (hsv[:,:,2] > 100)
        
        # Find contours for red units
        contours, _ = cv2.findContours(red_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    units.append({'xy': (cx, cy), 'belong': 'enemy'})
        
        # Find contours for blue units  
        contours, _ = cv2.findContours(blue_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    units.append({'xy': (cx, cy), 'belong': 'friendly'})
        
        return units
    
    def get_state(self, img: np.ndarray) -> Dict:
        """Extract full game state from screenshot."""
        self.elixir = self.extract_elixir(img)
        arena = self.extract_arena_features(img)
        units = self.detect_units_simple(img)
        
        return {
            'elixir': self.elixir,
            'arena': arena,  # (256, 128, 3) image
            'units': units,
            'time': self.time,
        }


class KataCRPerception:
    """
    KataCR perception using YOLOv8 directly with ultralytics.
    Detects all units on the arena with class labels.
    No JAX dependency - uses pretrained weights directly.
    """
    
    def __init__(self, conf: float = 0.5, iou: float = 0.5):
        """
        Initialize YOLOv8 detector with KataCR weights.
        
        Args:
            conf: Confidence threshold for detection
            iou: IOU threshold for NMS
        """
        self.conf = conf
        self.iou = iou
        self._init_models()
        self.state_builder = SimplifiedStateBuilder()
        
    def _init_models(self):
        """Initialize YOLOv8 models with KataCR weights."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not available. Run: pip install ultralytics==8.1.24")
        
        # Path to detector weights
        self.detector_paths = [
            KATACR_PATH / "runs" / "detector1_v0.7.13" / "best.pt",
            KATACR_PATH / "runs" / "detector2_v0.7.13" / "best.pt",
        ]
        
        # Check weights exist
        for p in self.detector_paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"Detector weights not found: {p}\n"
                    f"Download from: https://drive.google.com/file/d/1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_\n"
                    f"and: https://drive.google.com/file/d/1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD"
                )
        
        # PyTorch 2.6+ has stricter security for loading weights
        # We trust KataCR weights, so allow unsafe loading
        import torch
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        
        # Monkey-patch torch.load for ultralytics
        original_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_load
        
        try:
            # Load YOLO models
            self.detectors = [YOLO(str(p)) for p in self.detector_paths]
            print(f"✓ KataCR YOLOv8 detectors loaded (conf={self.conf}, iou={self.iou})")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
    
    def detect_units(self, img: np.ndarray) -> List[Dict]:
        """
        Detect all units in the arena using YOLOv8.
        
        Args:
            img: BGR image from ADB screenshot
            
        Returns:
            List of detected units with:
            - xyxy: Bounding box coordinates
            - conf: Detection confidence  
            - cls: Class index
            - name: Unit name (e.g., 'hog_rider', 'musketeer')
        """
        import torch
        import torchvision
        
        all_boxes = []
        
        # Run both detectors
        for detector in self.detectors:
            results = detector.predict(img, verbose=False, conf=self.conf)[0]
            boxes = results.boxes
            
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())
                    cls = int(boxes.cls[i].cpu())
                    
                    # Get unit name from detector's names or KataCR labels
                    if cls in results.names:
                        name = results.names[cls]
                    elif LABELS_AVAILABLE and cls in idx2unit:
                        name = idx2unit[cls]
                    else:
                        name = f"unit_{cls}"
                    
                    all_boxes.append({
                        'xyxy': xyxy.tolist(),
                        'conf': conf,
                        'cls': cls,
                        'name': name,
                    })
        
        # NMS across both detectors
        if len(all_boxes) > 1:
            boxes_tensor = torch.tensor([b['xyxy'] for b in all_boxes])
            confs_tensor = torch.tensor([b['conf'] for b in all_boxes])
            keep = torchvision.ops.nms(boxes_tensor, confs_tensor, self.iou)
            all_boxes = [all_boxes[i] for i in keep.tolist()]
        
        # Add center coordinates
        for box in all_boxes:
            x1, y1, x2, y2 = box['xyxy']
            box['center'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        return all_boxes
    
    def get_state(self, img: np.ndarray) -> Dict:
        """
        Extract full game state using YOLOv8 detection.
        
        Returns:
            Dict with:
            - elixir: Current elixir count (0-10)
            - arena: Resized arena image for model input
            - units: List of detected units with positions and types
            - unit_names: List of detected unit names
        """
        # Use simple elixir detection (reliable)
        elixir = self.state_builder.extract_elixir(img)
        arena = self.state_builder.extract_arena_features(img)
        
        # Use YOLOv8 for unit detection
        units = self.detect_units(img)
        
        # Separate by position (top half = enemy, bottom half = friendly)
        # Arena Y range: 580-1850, midpoint ~1215
        friendly_units = []
        enemy_units = []
        for u in units:
            cy = u['center'][1]
            if cy > 1215:  # Bottom half
                u['belong'] = 'friendly'
                friendly_units.append(u)
            else:  # Top half
                u['belong'] = 'enemy'
                enemy_units.append(u)
        
        return {
            'elixir': elixir,
            'arena': arena,
            'units': units,
            'friendly_units': len(friendly_units),
            'enemy_units': len(enemy_units),
            'unit_names': [u['name'] for u in units],
        }
    
    def visualize(self, img: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """Draw detection results on image."""
        units = self.detect_units(img)
        
        annotated = img.copy()
        for u in units:
            x1, y1, x2, y2 = [int(v) for v in u['xyxy']]
            color = (0, 255, 0) if u.get('belong') == 'friendly' else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{u['name']} {u['conf']:.2f}"
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if save_path:
            cv2.imwrite(save_path, annotated)
        
        return annotated


def create_perception_pipeline(use_full_katacr: bool = False) -> Tuple:
    """
    Create the perception pipeline.
    
    Args:
        use_full_katacr: If True, use KataCR's YOLOv8 combo detector.
                        If False, use simplified color-based perception.
    
    Returns:
        (screen_capture, state_builder)
    """
    capture = ADBScreenCapture()
    
    if use_full_katacr:
        try:
            perception = KataCRPerception(conf=0.5)
            return capture, perception
        except Exception as e:
            print(f"Could not initialize KataCR: {e}")
            print("Falling back to simplified perception")
    
    # Fall back to simplified
    state_builder = SimplifiedStateBuilder()
    return capture, state_builder


if __name__ == "__main__":
    # Test the perception pipeline
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--katacr", action="store_true", help="Use full KataCR detection")
    parser.add_argument("--save", type=str, default=None, help="Save annotated image")
    args = parser.parse_args()
    
    capture, state_builder = create_perception_pipeline(use_full_katacr=args.katacr)
    
    print("Testing ADB capture...")
    success, img = capture.read()
    
    if success:
        print(f"Captured image: {img.shape}")
        state = state_builder.get_state(img)
        print(f"Elixir: {state['elixir']}")
        
        if 'units' in state:
            print(f"Units detected: {len(state['units'])}")
            if args.katacr and state['units']:
                print("Unit names:", state.get('unit_names', []))
                print(f"Friendly: {state.get('friendly_units', 0)}, Enemy: {state.get('enemy_units', 0)}")
                
                # Visualize if requested
                if args.save and hasattr(state_builder, 'visualize'):
                    state_builder.visualize(img, args.save)
                    print(f"Saved annotated image to {args.save}")
        
        print(f"Arena shape: {state['arena'].shape}")
    else:
        print("Failed to capture screenshot")
