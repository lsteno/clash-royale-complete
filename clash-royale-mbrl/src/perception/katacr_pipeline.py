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
import time

import cv2
import numpy as np

# --- Wire KataCR on the path -------------------------------------------------
KATACR_ROOT = Path(__file__).resolve().parents[3] / "KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.build_dataset.constant import part3_elixir_params
from katacr.build_dataset.utils.split_part import extract_bbox, process_part
from katacr.constants.card_list import card2idx as DEFAULT_CARD2IDX
from katacr.ocr_text.paddle_ocr import OCR
from katacr.policy.perceptron.reward_builder import RewardBuilder
from katacr.policy.perceptron.state_builder import StateBuilder
from katacr.yolov8.combo_detect import ComboDetector


@dataclass
class KataCRVisionConfig:
    """Configuration for the KataCR perception stack."""

    detector_paths: Optional[List[Path]] = None
    detector_count: Optional[int] = None
    classifier_path: Optional[Path] = None
    enable_card_classifier: bool = True
    fallback_card_name: str = "skeletons"
    ocr_onnx: bool = False
    ocr_gpu: bool = True
    resize_width: int = 576   # KataCR canonical portrait width
    resize_height: int = 1280  # KataCR canonical portrait height
    enable_center_ocr: bool = True
    debug_save_parts: bool = False
    debug_parts_dir: Path = Path("logs/vision_parts")

    def resolved_detectors(self) -> List[Path]:
        if self.detector_paths is not None:
            paths = [Path(p) for p in self.detector_paths]
        else:
            paths = [
                KATACR_ROOT / "runs" / "detector1_v0.7.13" / "best.pt",
                KATACR_ROOT / "runs" / "detector2_v0.7.13" / "best.pt",
            ]
        if self.detector_count is not None:
            return paths[: max(1, int(self.detector_count))]
        return paths

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

    MAX_GAME_TIME = 360  # seconds (3 min + overtime buffer)
    MAX_EXTRAPOLATION_GAP = 5  # cap real-time inference gap in seconds

    def __init__(self, cfg: KataCRVisionConfig):
        self.cfg = cfg
        self.detectors = cfg.resolved_detectors()
        self.classifier_path = cfg.resolved_classifier()
        self._validate_assets()

        self.ocr = OCR(onnx=cfg.ocr_onnx, use_gpu=cfg.ocr_gpu, lang="en")
        self.yolo = ComboDetector(self.detectors)
        self.classifier = None
        if bool(getattr(cfg, "enable_card_classifier", True)):
            # KataCR's card classifier uses JAX/Flax and may be unavailable on some
            # machines. Keep it optional so the perception service can still run.
            from katacr.classification.predict import CardClassifier

            self.classifier = CardClassifier(self.classifier_path)
        self._last_time = 0
        self._last_capture_ts: Optional[float] = None
        self._last_elixir: Optional[int] = None

    def _validate_assets(self):
        missing = [p for p in self.detectors if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "KataCR detector weights missing: " + ", ".join(str(p) for p in missing)
            )
        if bool(getattr(self.cfg, "enable_card_classifier", True)) and not self.classifier_path.exists():
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
            # Nudge timer crop (part1) 35px left and 60px down to better align the HUD
            # (adjusted +15 right and +25 down for capture-region 1,38,494,1074)
            if i == 0 and isinstance(box_params, tuple) and len(box_params) == 4:
                img, box_params = self._shift_part_crop(frame_bgr, box_params, dx_px=-35, dy_px=60)
            parts.append(img)
            parts_pos.append(box_params)
        parts_pos = np.array(parts_pos)
        parts_pos = (parts_pos.reshape(-1, 2) * np.array(frame_bgr.shape[:2][::-1])).astype(np.int32).reshape(-1, 4)

        if self.cfg.debug_save_parts:
            self._save_debug_parts(frame_bgr, parts)

        now_ts = time.time()
        time_val = self.ocr.process_part1(parts[0], pil=False)
        ocr_time_failed = np.isinf(time_val)
        if ocr_time_failed:
            time_val = self._extrapolate_time(now_ts)
        self._last_time = min(time_val, self.MAX_GAME_TIME)
        self._last_capture_ts = now_ts
        arena = self.yolo.infer(parts[1], pil=False)
        if self.classifier is not None:
            cards = self.classifier.process_part3(parts[2], pil=False)
            card2idx = self.classifier.card2idx
            idx2card = self.classifier.idx2card
        else:
            fallback = str(getattr(self.cfg, "fallback_card_name", "skeletons"))
            cards = [fallback] * 5
            card2idx = DEFAULT_CARD2IDX
            idx2card = None
        elixir_raw = self.ocr.process_part3_elixir(parts[2], pil=False)
        elixir_fallback_used = False
        try:
            elixir = int(elixir_raw)
            elixir_failed = False
            self._last_elixir = elixir
        except (TypeError, ValueError):
            elixir_failed = True
            if self._last_elixir is not None:
                elixir = self._last_elixir
                elixir_fallback_used = True
            else:
                elixir = -1  # Still mark failure if we have no history
        
        # Check for Victory/Defeat/Match Over texts in the center (optional).
        center_flag = -1
        if bool(getattr(self.cfg, "enable_center_ocr", True)):
            center_flag = self.ocr.process_center_texts(frame_bgr, pil=False)

        return {
            "time": time_val,
            "ocr_time_failed": ocr_time_failed,
            "arena": arena,
            "cards": cards,
            "elixir": elixir,
            "elixir_raw": elixir_raw,
            "elixir_failed": elixir_failed,
            "elixir_fallback_used": elixir_fallback_used,
            "center_flag": center_flag,
            "card2idx": card2idx,
            "idx2card": idx2card,
            "parts_pos": parts_pos,
        }

    def _shift_part_crop(self, frame_bgr: np.ndarray, box_params: tuple, dx_px: int, dy_px: int):
        x, y, w, h = box_params
        fh, fw = frame_bgr.shape[:2]
        x_px, y_px = int(fw * x), int(fh * y)
        w_px, h_px = int(fw * w), int(fh * h)
        x_px = int(np.clip(x_px + dx_px, 0, max(0, fw - w_px)))
        y_px = int(np.clip(y_px + dy_px, 0, max(0, fh - h_px)))
        x1, y1 = x_px + w_px, y_px + h_px
        cropped = frame_bgr[y_px:y1, x_px:x1]
        # Update params back to proportions so downstream math stays consistent
        new_params = (x_px / fw, y_px / fh, w_px / fw, h_px / fh)
        return cropped, new_params

    def _extrapolate_time(self, now_ts: float) -> int:
        """Estimate match clock when OCR fails using wall-clock deltas."""
        if self._last_capture_ts is None:
            return self._last_time
        delta = max(0.0, now_ts - self._last_capture_ts)
        if delta <= 1e-3:
            return self._last_time
        delta_seconds = min(int(round(delta)), self.MAX_EXTRAPOLATION_GAP)
        return min(self._last_time + delta_seconds, self.MAX_GAME_TIME)

    def _save_debug_parts(self, frame_bgr: np.ndarray, parts: List[np.ndarray]) -> None:
        try:
            out_dir = Path(self.cfg.debug_parts_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(str(out_dir / f"frame_{ts}.png"), frame_bgr)
            for idx, p in enumerate(parts, start=1):
                if p is None or getattr(p, "size", 0) == 0:
                    continue
                cv2.imwrite(str(out_dir / f"frame_{ts}_part{idx}.png"), p)
            # Save the elixir sub-crop from part3 for OCR debugging
            if len(parts) >= 3 and parts[2] is not None and getattr(parts[2], "size", 0) > 0:
                elixir_crop = extract_bbox(parts[2], *part3_elixir_params)
                if elixir_crop is not None and getattr(elixir_crop, "size", 0) > 0:
                    cv2.imwrite(str(out_dir / f"frame_{ts}_elixir.png"), elixir_crop)
        except Exception as exc:
            print(f"[VisionDebug] failed to save parts: {exc}")


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
