#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class FrameSummary:
    name: str
    frame_id: Optional[int]
    want_action: Optional[bool]
    boxes: int
    unit_infos: int
    label_counts: Counter
    match_over: Optional[bool]


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _iter_dump_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("frame_")])


def _is_structure_or_ui(label: str) -> bool:
    # KataCR labels include bel suffix for unit/state (0/1). Bars are part of HP overlays.
    prefixes = (
        "king-tower",
        "queen-tower",
        "tower-bar",
        "bar",
        "bar-level",
        "clock",
        "elixir",
    )
    return label.startswith(prefixes)


def summarize_dir(d: Path) -> FrameSummary:
    meta = _load_json(d / "meta.json")
    arena = _load_json(d / "arena_boxes.json")
    state = _load_json(d / "state.json")

    labels: List[str] = []
    if isinstance(arena, list):
        for row in arena:
            if isinstance(row, dict) and isinstance(row.get("label"), str):
                labels.append(row["label"])
    counts = Counter(labels)

    frame_id = meta.get("frame_id") if isinstance(meta, dict) else None
    want_action = meta.get("want_action") if isinstance(meta, dict) else None
    match_over = None
    if isinstance(meta, dict):
        match_over = bool(meta.get("done", False))

    unit_infos = -1
    if isinstance(state, dict):
        unit_infos = len(state.get("unit_infos", []) or [])

    return FrameSummary(
        name=d.name,
        frame_id=int(frame_id) if isinstance(frame_id, int) else None,
        want_action=bool(want_action) if isinstance(want_action, bool) else None,
        boxes=len(labels),
        unit_infos=int(unit_infos),
        label_counts=counts,
        match_over=match_over,
    )


def _structure_presence_ok(counts: Counter) -> bool:
    # Expect at least 6 tower bodies and 4 tower bars in a valid in-battle arena crop.
    needed = {
        "king-tower0": 1,
        "king-tower1": 1,
        "queen-tower0": 2,
        "queen-tower1": 2,
    }
    for k, v in needed.items():
        if counts.get(k, 0) < v:
            return False
    if counts.get("tower-bar0", 0) < 2:
        return False
    if counts.get("tower-bar1", 0) < 2:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate perception debug dumps")
    ap.add_argument("dump_dir", type=Path, help="Directory containing frame_* dump folders")
    ap.add_argument("--limit", type=int, default=25, help="How many frames to scan")
    args = ap.parse_args()

    dirs = _iter_dump_dirs(args.dump_dir)[: max(0, int(args.limit))]
    if not dirs:
        raise SystemExit(f"No frame_* dirs found under {args.dump_dir}")

    frames = [summarize_dir(d) for d in dirs]

    all_labels = Counter()
    valid = 0
    likely_end_screen = 0
    frames_with_any_units = 0
    frames_with_body_units = 0

    for f in frames:
        all_labels.update(f.label_counts)
        if _structure_presence_ok(f.label_counts):
            valid += 1
        elif f.boxes <= 2:
            likely_end_screen += 1
        if f.unit_infos and f.unit_infos > 6:
            frames_with_any_units += 1
        body = [k for k in f.label_counts if not _is_structure_or_ui(k)]
        if sum(f.label_counts[k] for k in body) > 0:
            frames_with_body_units += 1

    print(f"Scanned frames: {len(frames)}")
    print(f"Valid arena (towers/bars present): {valid}/{len(frames)}")
    print(f"Likely non-battle UI (<=2 boxes): {likely_end_screen}/{len(frames)}")
    print(f"Frames with unit_infos > 6: {frames_with_any_units}/{len(frames)}")
    print(f"Frames with any body-unit labels: {frames_with_body_units}/{len(frames)}")

    print("\nTop labels:")
    for k, v in all_labels.most_common(20):
        print(f"  {k}: {v}")

    print("\nPer-frame summary:")
    for f in frames:
        body = [k for k in f.label_counts if not _is_structure_or_ui(k)]
        body_n = sum(f.label_counts[k] for k in body)
        ok = _structure_presence_ok(f.label_counts)
        print(
            f"{f.name} frame_id={f.frame_id} want_action={f.want_action} "
            f"boxes={f.boxes} unit_infos={f.unit_infos} body_units={body_n} valid={ok}"
        )


if __name__ == "__main__":
    main()

