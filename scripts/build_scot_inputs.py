#!/usr/bin/env python3
"""Write SCoT validation questions as an input JSONL for scripts/run_inference.py.

Object list, scene video, and top-down image, the default input (all six tasks):
    python scripts/build_scot_inputs.py --per-task 30 \\
        --video-dir data/raw/scene_frames --image-dir data/raw/scene_images \\
        --out data/processed/pilot_full.jsonl

Numbered object boxes drawn on the top-down image, one image per question:
    python scripts/build_scot_inputs.py --per-task 30 --tasks implicit_detection \\
        --video-dir data/raw/scene_frames --image-dir data/raw/scene_images --mark-objects \\
        --out data/processed/pilot_marked.jsonl

Object list only, a comparison with less information:
    python scripts/build_scot_inputs.py --per-task 30 --out data/processed/pilot_text.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from PIL import Image

from thinkenough.scot import (
    VAL_FILES,
    alignment_is_usable,
    draw_object_map,
    fetch_scene_boxes,
    fetch_scene_frames,
    fetch_scene_images,
    fit_scene_alignment,
    load_attributes,
    load_val_records,
    make_samples,
)


def mark_images(samples: list, mask3d: dict, image_dir: Path) -> None:
    """Replace each sample's image with one showing its shuffled object ids."""
    human = load_attributes("scannet_val_attributes.pt")
    boxes_dir = image_dir.parent / "scene_boxes"
    marked_dir = image_dir.parent / "marked_images"
    marked_dir.mkdir(parents=True, exist_ok=True)
    fetch_scene_boxes((s.metadata["scene_id"] for s in samples), boxes_dir)

    alignments: dict[str, dict | None] = {}
    unmarked = 0
    for sample in samples:
        scene_id = sample.metadata["scene_id"]
        if scene_id not in alignments:
            rows = json.loads((boxes_dir / f"{scene_id}_bbox.json").read_text(encoding="utf-8"))["bboxes"]
            alignments[scene_id] = fit_scene_alignment([r["bbox"] for r in rows], human[scene_id]["locs"])
        alignment = alignments[scene_id]
        sample.metadata["image_marked"] = alignment_is_usable(alignment)
        if not sample.metadata["image_marked"]:
            # Keep the plain image rather than draw boxes in the wrong place.
            unmarked += 1
            continue
        attrs = mask3d[scene_id]
        with Image.open(_REPO_ROOT / sample.image_path) as image:
            marked = draw_object_map(
                image.convert("RGB"), attrs["objects"], attrs["locs"], sample.metadata["object_order"], alignment
            )
        path = marked_dir / f"{sample.sample_id}.jpg"
        marked.save(path, format="JPEG", quality=92)
        sample.image_path = str(path.relative_to(_REPO_ROOT))
    print(f"marked {len(samples) - unmarked} images; {unmarked} kept plain because alignment failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SCoT input JSONL")
    parser.add_argument("--tasks", nargs="+", choices=sorted(VAL_FILES), default=sorted(VAL_FILES))
    parser.add_argument("--per-task", type=int, default=30)
    parser.add_argument("--split", choices=["tune", "final"], default="tune")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--image-dir", default=None, help="Attach and download scene images here")
    parser.add_argument("--video-dir", default=None, help="Attach and extract video frames here")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--mark-objects", action="store_true", help="Draw numbered boxes; needs --image-dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mark_objects and not args.image_dir:
        parser.error("--mark-objects needs --image-dir")

    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    samples = []
    for task in args.tasks:
        chosen = make_samples(
            task,
            load_val_records(task),
            mask3d,
            per_task=args.per_task,
            seed=args.seed,
            split=args.split,
            image_dir=args.image_dir,
            video_dir=args.video_dir,
        )
        print(f"{task}: {len(chosen)} questions from {len({s.metadata['scene_id'] for s in chosen})} scenes")
        samples.extend(chosen)

    if args.image_dir:
        fetch_scene_images((s.metadata["scene_id"] for s in samples), _REPO_ROOT / args.image_dir)
    if args.mark_objects:
        mark_images(samples, mask3d, _REPO_ROOT / args.image_dir)

    if args.video_dir:
        fetch_scene_frames(
            (s.metadata["scene_id"] for s in samples),
            _REPO_ROOT / args.video_dir,
            args.num_frames,
        )

    out = _REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
