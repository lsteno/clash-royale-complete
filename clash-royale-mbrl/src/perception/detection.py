"""
YOLOv8 Detection Pipeline for Clash Royale.
Optimized for Apple Silicon MPS backend.

This module handles:
- Unit detection (friendly/enemy troops, buildings)
- Tower health estimation
- Card recognition
- Elixir bar reading
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not installed. Run: pip install ultralytics")

from ..utils.device import get_device


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized 0-1)
    center: Tuple[float, float]  # center x, y (normalized 0-1)
    is_friendly: bool = None  # True = friendly, False = enemy, None = unknown


@dataclass
class DetectionConfig:
    """Configuration for detection pipeline."""
    model_path: Optional[str] = None  # Path to YOLOv8 weights
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "mps"  # mps for Apple Silicon, cuda, or cpu
    img_size: int = 640


# Clash Royale unit categories (based on KataCR)
CR_CLASSES = {
    # Troops - Friendly and Enemy versions
    "knight": 0, "archers": 1, "giant": 2, "skeleton_army": 3,
    "minions": 4, "balloon": 5, "witch": 6, "barbarians": 7,
    "golem": 8, "hog_rider": 9, "valkyrie": 10, "musketeer": 11,
    "mini_pekka": 12, "wizard": 13, "royal_giant": 14, "prince": 15,
    "baby_dragon": 16, "skeleton_king": 17, "electro_wizard": 18,
    "bandit": 19, "inferno_dragon": 20, "lumberjack": 21,
    "mega_knight": 22, "sparky": 23, "miner": 24, "princess": 25,
    "ice_wizard": 26, "graveyard": 27, "lava_hound": 28,
    # Buildings
    "cannon": 50, "tesla": 51, "inferno_tower": 52, "bomb_tower": 53,
    "xbow": 54, "mortar": 55, "elixir_collector": 56, "goblin_hut": 57,
    # Spells (visual indicators)
    "fireball": 80, "arrows": 81, "rocket": 82, "lightning": 83,
    "freeze": 84, "poison": 85, "zap": 86, "tornado": 87,
    # Towers
    "king_tower": 100, "princess_tower": 101,
}


class ClashRoyaleDetector:
    """
    YOLOv8-based object detector for Clash Royale.
    Optimized for MPS (Apple Silicon) inference.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.device = get_device(self.config.device)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model."""
        if YOLO is None:
            raise ImportError("ultralytics package not installed")
        
        if self.config.model_path and Path(self.config.model_path).exists():
            # Load custom trained model
            self.model = YOLO(self.config.model_path)
        else:
            # Load pretrained YOLOv8 (will need fine-tuning)
            print("Loading pretrained YOLOv8n (will need fine-tuning for CR)")
            self.model = YOLO("yolov8n.pt")
        
        # Move to device
        # Note: ultralytics handles device management internally
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a frame.
        
        Args:
            frame: RGB image array (H, W, 3)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=str(self.device),
            verbose=False
        )
        
        detections = []
        h, w = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy / np.array([w, h, w, h])
                
                # Get class and confidence
                cls_id = int(boxes.cls[i].cpu())
                conf = float(boxes.conf[i].cpu())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                
                # Calculate center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Determine if friendly (bottom half of arena = friendly)
                is_friendly = cy > 0.5
                
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    is_friendly=is_friendly
                ))
        
        return detections


class StateGridBuilder:
    """
    Converts YOLO detections into a multi-channel grid tensor
    suitable for DreamerV3 input.
    
    Grid: (32, 18, N) where N channels represent:
    - Friendly units (density/presence)
    - Enemy units (density/presence)
    - Building HP
    - Elixir state
    """
    
    GRID_WIDTH = 32
    GRID_HEIGHT = 18
    NUM_CHANNELS = 8  # Configurable based on needs
    
    # Channel indices
    CH_FRIENDLY_UNITS = 0
    CH_ENEMY_UNITS = 1
    CH_FRIENDLY_BUILDINGS = 2
    CH_ENEMY_BUILDINGS = 3
    CH_TOWER_HP = 4
    CH_ELIXIR = 5
    CH_SPELL_ZONES = 6
    CH_CARD_READY = 7
    
    def __init__(self):
        self.grid = np.zeros((self.NUM_CHANNELS, self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
    
    def reset(self):
        """Clear the grid."""
        self.grid.fill(0)
    
    def _position_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized (0-1) position to grid cell."""
        gx = min(int(x * self.GRID_WIDTH), self.GRID_WIDTH - 1)
        gy = min(int(y * self.GRID_HEIGHT), self.GRID_HEIGHT - 1)
        return gx, gy
    
    def build(self, detections: List[Detection], elixir: int = 10, 
              tower_hp: Dict[str, float] = None) -> np.ndarray:
        """
        Build state grid from detections.
        
        Args:
            detections: List of Detection objects
            elixir: Current elixir count (0-10)
            tower_hp: Dict mapping tower names to HP percentage (0-1)
            
        Returns:
            Grid tensor (NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH)
        """
        self.reset()
        
        for det in detections:
            gx, gy = self._position_to_grid(det.center[0], det.center[1])
            
            # Determine channel based on type and allegiance
            if "tower" in det.class_name.lower():
                # Tower
                channel = self.CH_TOWER_HP
                self.grid[channel, gy, gx] = det.confidence  # Use confidence as proxy for HP
            elif det.is_friendly:
                # Friendly unit/building
                if det.class_id in range(50, 60):  # Buildings
                    channel = self.CH_FRIENDLY_BUILDINGS
                else:
                    channel = self.CH_FRIENDLY_UNITS
                self.grid[channel, gy, gx] += det.confidence
            else:
                # Enemy unit/building
                if det.class_id in range(50, 60):  # Buildings
                    channel = self.CH_ENEMY_BUILDINGS
                else:
                    channel = self.CH_ENEMY_UNITS
                self.grid[channel, gy, gx] += det.confidence
        
        # Set elixir channel (uniform across grid)
        self.grid[self.CH_ELIXIR, :, :] = elixir / 10.0
        
        return self.grid.copy()
    
    def to_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Convert grid to PyTorch tensor."""
        if device is None:
            device = get_device()
        return torch.from_numpy(self.grid).unsqueeze(0).to(device)


class PerceptionPipeline:
    """
    Full perception pipeline: Frame -> Detections -> State Grid.
    Integrates YOLO detection with state abstraction.
    """
    
    def __init__(self, detection_config: Optional[DetectionConfig] = None):
        self.detector = ClashRoyaleDetector(detection_config)
        self.grid_builder = StateGridBuilder()
        self._last_detections: List[Detection] = []
    
    def process(self, frame: np.ndarray, elixir: int = 10) -> Tuple[np.ndarray, List[Detection]]:
        """
        Process a single frame through the perception pipeline.
        
        Args:
            frame: RGB image (H, W, 3)
            elixir: Current elixir count
            
        Returns:
            Tuple of (state_grid, detections)
        """
        # Run detection
        self._last_detections = self.detector.detect(frame)
        
        # Build state grid
        state_grid = self.grid_builder.build(self._last_detections, elixir=elixir)
        
        return state_grid, self._last_detections
    
    def process_to_tensor(self, frame: np.ndarray, elixir: int = 10,
                          device: torch.device = None) -> torch.Tensor:
        """Process frame and return as PyTorch tensor."""
        state_grid, _ = self.process(frame, elixir)
        if device is None:
            device = get_device()
        return torch.from_numpy(state_grid).unsqueeze(0).to(device)
    
    @property
    def last_detections(self) -> List[Detection]:
        """Get detections from last processed frame."""
        return self._last_detections


if __name__ == "__main__":
    # Test detection pipeline
    print("Testing Clash Royale Detection Pipeline...")
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    # Initialize pipeline
    config = DetectionConfig(device="mps")
    pipeline = PerceptionPipeline(config)
    
    # Process frame
    state_grid, detections = pipeline.process(frame, elixir=7)
    
    print(f"State grid shape: {state_grid.shape}")
    print(f"Number of detections: {len(detections)}")
    print(f"Device: {get_device()}")
