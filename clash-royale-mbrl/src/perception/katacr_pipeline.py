"""
KataCR perception bridge for the macOS emulator loop.

Responsibilities:
- Normalize emulator frames to the canonical KataCR size.
- Run the VisualFusion stack (OCR + ComboDetector + CardClassifier).
- Build game state via KataCR StateBuilder.
- Compute rewards via KataCR RewardBuilder.

This isolates all KataCR-specific wiring so the gym/env wrapper can stay thin.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import cv2
import numpy as np

# --- Wire KataCR on the path -------------------------------------------------
KATACR_ROOT = Path(__file__).resolve().parents[3] / "KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.build_dataset.utils.split_part import process_part
from katacr.classification.predict import CardClassifier
from katacr.ocr_text.paddle_ocr import OCR
from katacr.policy.perceptron.reward_builder import RewardBuilder
from katacr.policy.perceptron.state_builder import StateBuilder
from katacr.yolov8.combo_detect import ComboDetector


@dataclass
class KataCRVisionConfig:
    """Configuration for the KataCR perception stack."""

    detector_paths: Optional[List[Path]] = None
    classifier_path: Optional[Path] = None
    ocr_onnx: bool = False
    ocr_gpu: bool = True
    resize_width: int = 1280  # KataCR canonical width
    resize_height: int = 576  # KataCR canonical height

    def resolved_detectors(self) -> List[Path]:
        if self.detector_paths is not None:
            return [Path(p) for p in self.detector_paths]
        return [
            KATACR_ROOT / "runs" / "detector1_v0.7.13" / "best.pt",
            KATACR_ROOT / "runs" / "detector2_v0.7.13" / "best.pt",
        ]

    def resolved_classifier(self) -> Path:
        if self.classifier_path is not None:
            return Path(self.classifier_path)
        return KATACR_ROOT / "logs" / "CardClassification-checkpoints"


@dataclass
class KataCRPerceptionResult:
    state: Dict
    reward: float
    info: Dict


class VisualFusionAdapter:
    """
    Minimal reimplementation of KataCR VisualFusion with configurable weights.

    It keeps the original behavior (OCR + detector + card classifier) but lets
    us point to the correct local weight files.
    """

    def __init__(self, cfg: KataCRVisionConfig):
        self.cfg = cfg
        self.detectors = cfg.resolved_detectors()
        self.classifier_path = cfg.resolved_classifier()
        self._validate_assets()

        self.ocr = OCR(onnx=cfg.ocr_onnx, use_gpu=cfg.ocr_gpu, lang="en")
        self.yolo = ComboDetector(self.detectors)
        self.classifier = CardClassifier(self.classifier_path)

    def _validate_assets(self):
        missing = [p for p in self.detectors if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "KataCR detector weights missing: " + ", ".join(str(p) for p in missing)
            )
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                "KataCR card classifier weights missing. Download CardClassification-checkpoints "
                "from the KataCR README link and place under KataCR/logs/."
            )

    def process(self, frame_bgr: np.ndarray) -> Dict:
        parts = []
        parts_pos = []
        # Parts: 1 = time HUD, 2 = arena, 3 = cards/elixir
        for i in range(3):
            img, box_params = process_part(frame_bgr, i + 1, verbose=True)
            parts.append(img)
            parts_pos.append(box_params)
        parts_pos = np.array(parts_pos)
        parts_pos = (parts_pos.reshape(-1, 2) * np.array(frame_bgr.shape[:2][::-1])).astype(np.int32).reshape(-1, 4)

        time_val = self.ocr.process_part1(parts[0], pil=False)
        arena = self.yolo.infer(parts[1], pil=False)
        cards = self.classifier.process_part3(parts[2], pil=False)
        elixir = self.ocr.process_part3_elixir(parts[2], pil=False)

        return {
            "time": time_val,
            "arena": arena,
            "cards": cards,
            "elixir": elixir,
            "card2idx": self.classifier.card2idx,
            "idx2card": self.classifier.idx2card,
            "parts_pos": parts_pos,
        }


class KataCRPerceptionEngine:
    """End-to-end KataCR perception: frame -> state -> reward."""

    def __init__(self, cfg: Optional[KataCRVisionConfig] = None):
        self.cfg = cfg or KataCRVisionConfig()
        self.visual = VisualFusionAdapter(self.cfg)
        self.state_builder = StateBuilder()
        self.reward_builder = RewardBuilder()

    def reset(self):
        self.state_builder.reset()
        self.reward_builder.reset()

    def _resize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[1] == self.cfg.resize_width and frame_bgr.shape[0] == self.cfg.resize_height:
            return frame_bgr
        return cv2.resize(
            frame_bgr,
            (self.cfg.resize_width, self.cfg.resize_height),
            interpolation=cv2.INTER_AREA,
        )

    def process(
        self,
        frame_bgr: np.ndarray,
        deploy_cards: Optional[Iterable[str]] = None,
    ) -> KataCRPerceptionResult:
        frame_bgr = self._resize_frame(frame_bgr)
        info = self.visual.process(frame_bgr)

        deploy_cards_set: Set[str] = set(deploy_cards) if deploy_cards is not None else set()
        self.state_builder.update(info, deploy_cards_set)
        self.reward_builder.update(info)

        state = self.state_builder.get_state()
        reward = self.reward_builder.get_reward()
        return KataCRPerceptionResult(state=state, reward=reward, info=info)

