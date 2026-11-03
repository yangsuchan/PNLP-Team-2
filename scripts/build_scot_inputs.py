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

Keep that input and add a numbered map and each object's estimated color, so the model need not
link video to object ids itself (run scripts/build_object_colors.py first):
    python scripts/build_scot_inputs.py --per-task 30 --tasks implicit_detection \\
        --video-dir data/raw/scene_frames --image-dir data/raw/scene_images \\
        --add-map --colors data/processed/object_colors.json --out data/processed/pilot_linked.jsonl

Object list only, a comparison with less information:
    python scripts/build_scot_inputs.py --per-task 30 --out data/processed/pilot_text.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from PIL import Image

from thinkenough.scot import (
    OBJECT_MAP_NOTE,
    VAL_FILES,
    asks_distance,
    clean_scene_id,
    draw_object_map,
    fetch_scene_boxes,
    fetch_scene_frames,
    fetch_scene_images,
    fit_scene_alignment,
    load_attributes,
    load_val_records,
    make_samples,
)


def fit_alignments(scene_ids: set[str], boxes_dir: Path, human: dict) -> dict:
    """Fit the scan-frame alignment of each scene from SCoT's box files."""
    fetch_scene_boxes(scene_ids, boxes_dir)
    alignments = {}
    for scene_id in sorted(scene_ids):
        rows = json.loads((boxes_dir / f"{scene_id}_bbox.json").read_text(encoding="utf-8"))["bboxes"]
        alignments[scene_id] = fit_scene_alignment([r["bbox"] for r in rows], human[scene_id]["locs"])
    return alignments


def mark_images(samples: list, mask3d: dict, alignments: dict, image_dir: Path) -> None:
    """Replace each sample's image with one showing its shuffled object ids."""
    marked_dir = image_dir.parent / "marked_images"
    marked_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        scene_id = sample.metadata["scene_id"]
        attrs = mask3d[scene_id]
        with Image.open(_REPO_ROOT / sample.image_path) as image:
            marked = draw_object_map(
                image.convert("RGB"), attrs["objects"], attrs["locs"], sample.metadata["object_order"], alignments[scene_id]
            )
        path = marked_dir / f"{sample.sample_id}.jpg"
        marked.save(path, format="JPEG", quality=92)
        sample.image_path = str(path.relative_to(_REPO_ROOT))
        sample.metadata["image_marked"] = True


def add_object_maps(samples: list, mask3d: dict, alignments: dict, image_dir: Path) -> None:
    """Add a numbered map per question after its plain image, which stays."""
    map_dir = image_dir.parent / "marked_images"
    map_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        meta = sample.metadata
        attrs = mask3d[meta["scene_id"]]
        with Image.open(_REPO_ROOT / sample.image_path) as image:
            marked = draw_object_map(
                image.convert("RGB"), attrs["objects"], attrs["locs"], meta["object_order"], alignments[meta["scene_id"]]
            )
        path = map_dir / f"{sample.sample_id}_map.jpg"
        marked.save(path, format="JPEG", quality=92)
        sample.extra_image_paths.append(str(path.relative_to(_REPO_ROOT)))
        cut = -len(meta["prompt"])
        sample.question = f"{sample.question[:cut]}{OBJECT_MAP_NOTE}\n\n{sample.question[cut:]}"
        meta["object_map"] = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SCoT input JSONL")
    parser.add_argument("--tasks", nargs="+", choices=sorted(VAL_FILES), default=sorted(VAL_FILES))
    parser.add_argument("--per-task", type=int, default=30)
    parser.add_argument("--split", choices=["tune", "final"], default="tune")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--image-dir", default=None, help="Attach and download scene images here")
    parser.add_argument("--video-dir", default=None, help="Attach and extract video frames here")
    parser.add_argument("--num-frames", type=int, default=None, help="Default: 2 frames per second, the model card's default")
    parser.add_argument("--boxes-dir", default="data/raw/scene_boxes", help="SCoT box files are saved here")
    parser.add_argument("--object-list", choices=["mask3d", "human"], default="mask3d")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep object ids in file order")
    parser.add_argument("--mark-objects", action="store_true", help="Draw numbered boxes; needs --image-dir")
    parser.add_argument("--add-map", action="store_true", help="Add a numbered map after the plain image; needs --image-dir")
    parser.add_argument("--colors", default=None, help="JSON of estimated colors by scene and object, Mask3D list only")
    parser.add_argument("--distance-only", action="store_true", help="Keep questions asking a distance, the ones scored from coordinates")
    parser.add_argument("--mixed-order", action="store_true", help="Run questions in random order, so a stopped run is still a random sample")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mark_objects and not args.image_dir:
        parser.error("--mark-objects needs --image-dir")
    if args.mark_objects and args.object_list != "mask3d":
        parser.error("--mark-objects draws the Mask3D list only")
    if args.add_map and (not args.image_dir or args.mark_objects or args.object_list != "mask3d"):
        parser.error("--add-map needs --image-dir and the Mask3D list, without --mark-objects")
    if args.colors and args.object_list != "mask3d":
        parser.error("--colors describes the Mask3D list only")

    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    human = load_attributes("scannet_val_attributes.pt")
    records = {task: load_val_records(task) for task in args.tasks}
    scene_ids = {clean_scene_id(r["scene_id"]) for rows in records.values() for r in rows}
    alignments = fit_alignments(scene_ids, _REPO_ROOT / args.boxes_dir, human)
    samples = []
    for task in args.tasks:
        chosen = make_samples(
            task,
            records[task],
            human if args.object_list == "human" else mask3d,
            per_task=args.per_task,
            seed=args.seed,
            split=args.split,
            image_dir=args.image_dir,
            video_dir=args.video_dir,
            shuffle_ids=not args.no_shuffle,
            alignments=alignments,
            object_list=args.object_list,
            colors=json.loads((_REPO_ROOT / args.colors).read_text(encoding="utf-8")) if args.colors else None,
            keep=asks_distance if args.distance_only else None,
        )
        print(f"{task}: {len(chosen)} questions from {len({s.metadata['scene_id'] for s in chosen})} scenes")
        samples.extend(chosen)

    if args.image_dir:
        fetch_scene_images((s.metadata["scene_id"] for s in samples), _REPO_ROOT / args.image_dir)
    if args.mark_objects:
        mark_images(samples, mask3d, alignments, _REPO_ROOT / args.image_dir)
    if args.add_map:
        add_object_maps(samples, mask3d, alignments, _REPO_ROOT / args.image_dir)

    if args.video_dir:
        fetch_scene_frames(
            (s.metadata["scene_id"] for s in samples),
            _REPO_ROOT / args.video_dir,
            args.num_frames,
        )

    if args.mixed_order:
        random.Random(args.seed).shuffle(samples)

    out = _REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
