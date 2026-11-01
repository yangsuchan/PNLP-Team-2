"""SCoT Level 2-3 validation data and metrics for Qwen3-VL.

Questions, references, and metrics follow SCoT unchanged. Only the input differs:
SCoT-Reasoner reads trained object features from the scan and video, while Qwen
reads the same Mask3D objects as text and watches frames of the same scene video.
A top-down image from SCoT's Drive is available as a further input.
"""

from __future__ import annotations

import colorsys
import hashlib
import html
import io
import json
import math
import random
import re
import shutil
import statistics
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from thinkenough.schemas import InferenceSample

SCOT_REPO = "Orange-zZZ/SCoT"
SCENE_REPO = "ZzZZCHS/Chat-Scene"
IMAGE_FOLDER_ID = "1_4L8uDdSG4ykFwcZjgEbo-uvCprJ5hWP"  # SCoT "source data" Drive folder
VIDEO_REPO = "nyu-visionx/VSI-Bench"  # scannet.zip holds a video for each of 312 val scenes

VAL_FILES = {
    "implicit_detection": "SCoT_Val/ImplicitDetection_Val.json",
    "object_analysis": "SCoT_Val/ObjectAnalysis_Val.json",
    "relationship_analysis": "SCoT_Val/RelationshipAnalysis_Val.json",
    "scene_analysis": "SCoT_Val/SceneAnalysis_Val.json",
    "situated_planning": "SCoT_Val/SituatedPlanning_Val.json",
    "unsituated_planning": "SCoT_Val/UnsituatedPlanning_Val.json",
}

IMAGE_METERS_PER_PIXEL = 0.005  # From SCoT's utils/Eval_SCoT_LLM_Score.py.
UNDRAWN_OBJECTS = {"ceiling", "floor", "wall"}
MIN_ALIGNMENT_PAIRS = 5
MAX_ALIGNMENT_ERROR = 0.1  # Meters.

# Wording from SCoT's prompts/system.txt and prompts/instruction.txt.
SCOT_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SCENE_INSTRUCTION = (
    "The conversation centers around an indoor scene. Object information: [\n{objects}\n]."
)

_OBJECT_ID = re.compile(r"<?OBJ(\d{3})>?")
_COORDINATES = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_METERS = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:m\b|meters?\b|metres?\b)", re.IGNORECASE)


def clean_scene_id(scene_id: str) -> str:
    """Remove the trailing comma that some SCoT train files keep in scene ids."""
    return scene_id.rstrip(",").strip()


def scene_split(scene_id: str) -> str:
    """Assign a scene to "tune" or "final" by hash, the same way for every task."""
    return "tune" if int(hashlib.sha1(scene_id.encode()).hexdigest(), 16) % 2 == 0 else "final"


def scene_text(objects: list[str], locs: Any, order: list[int] | None = None) -> str:
    """Render one scene's objects with SCoT's ``<OBJ###>`` ids.

    One decimal matches SCoT's own scene text and keeps 150-object lists shorter.

    Args:
        objects: Object names.
        locs: N x 6 values per object: center x, y, z and size x, y, z.
        order: Original object index shown at each id. SCoT also shuffles ids per
            question, so an id says nothing about the object. Default: file order.
    """
    rows = _as_rows(locs)
    lines = []
    for shown, original in enumerate(order if order is not None else range(len(rows))):
        cx, cy, cz, sx, sy, sz = rows[original]
        name = objects[original]
        lines.append(
            f"<OBJ{shown:03d}> {name}: center ({cx:.1f}, {cy:.1f}, {cz:.1f}), "
            f"size ({sx:.1f}, {sy:.1f}, {sz:.1f})"
        )
    return "\n".join(lines)


def build_question(
    prompt: str, objects: list[str], locs: Any, order: list[int] | None = None
) -> str:
    """Put SCoT's system line and the object list before an unchanged question."""
    scene = SCENE_INSTRUCTION.format(objects=scene_text(objects, locs, order))
    return f"{SCOT_SYSTEM} {scene}\n\n{prompt}"


def make_samples(
    task: str,
    records: list[dict[str, Any]],
    mask3d: dict[str, dict[str, Any]],
    *,
    per_task: int,
    seed: int = 1234,
    split: str = "tune",
    image_dir: str | None = None,
    video_dir: str | None = None,
) -> list[InferenceSample]:
    """Sample questions from one split. The same seed gives the same questions.

    Args:
        task: Key of ``VAL_FILES``.
        records: Parsed SCoT validation file.
        mask3d: Mask3D attributes by scene id.
        per_task: Number of questions to keep.
        seed: Sampling seed.
        split: "tune" or "final" from ``scene_split``.
        image_dir: If given, attach ``<image_dir>/<scene_id>.png`` to each sample.
        video_dir: If given, attach frames in ``<video_dir>/<scene_id>/`` to each sample.

    Returns:
        Samples whose metadata holds everything scoring needs.
    """
    pool = [
        (index, record)
        for index, record in enumerate(records)
        if scene_split(clean_scene_id(record["scene_id"])) == split
    ]
    chosen = sorted(
        random.Random(f"{seed}-{task}").sample(pool, min(per_task, len(pool))),
        key=lambda pair: pair[0],
    )

    samples = []
    for index, record in chosen:
        scene_id = clean_scene_id(record["scene_id"])
        attrs = mask3d[scene_id]
        # Names can outnumber boxes in the Mask3D file; SCoT counts objects by boxes too.
        order = list(range(len(_as_rows(attrs["locs"]))))
        random.Random(f"{seed}-{task}-{index}").shuffle(order)
        samples.append(
            InferenceSample(
                sample_id=f"{task}_{index:05d}",
                image_path=f"{image_dir}/{scene_id}.png" if image_dir else "",
                video_path=f"{video_dir}/{scene_id}" if video_dir else "",
                question=build_question(record["prompt"], attrs["objects"], attrs["locs"], order),
                metadata={
                    "task": task,
                    "split": split,
                    "scene_id": scene_id,
                    "record_index": index,
                    "obj_id": record.get("obj_id"),
                    "prompt": record["prompt"],
                    "references": record.get("ref_captions", []),
                    "object_order": order,
                },
            )
        )
    return samples


def load_val_records(task: str) -> list[dict[str, Any]]:
    """Download (or read from cache) one SCoT validation file."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SCOT_REPO, VAL_FILES[task], repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_attributes(name: str) -> dict[str, dict[str, Any]]:
    """Load scene attributes, e.g. ``scannet_mask3d_val_attributes.pt``."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SCENE_REPO, f"annotations/{name}", repo_type="dataset")
    return torch.load(path, map_location="cpu", weights_only=True)


def fetch_scene_images(scene_ids: Iterable[str], out_dir: str | Path) -> None:
    """Download SCoT scene images as PNG from its public Drive folder.

    Existing images are skipped. SCoT stores TIFF, which the Qwen processor cannot
    decode from a path, so each file is converted.

    Raises:
        FileNotFoundError: A scene is missing from the Drive listing.
        ValueError: Drive returned something other than a TIFF file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in sorted(set(scene_ids)) if not (out / f"{s}.png").is_file()]
    if not wanted:
        return

    files = _list_drive_subfolder("scans_tif")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    for scene_id in wanted:
        url = files.get(f"{scene_id}.tif")
        if url is None:
            raise FileNotFoundError(f"{scene_id}.tif is not in the SCoT Drive folder")
        file_id = re.search(r"/d/([^/?#]+)|id=([^&]+)", url)
        body = _read_drive_file(opener, file_id.group(1) or file_id.group(2))
        if body[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise ValueError(f"Drive did not return a TIFF for {scene_id}")
        partial = out / f"{scene_id}.png.partial"
        Image.open(io.BytesIO(body)).convert("RGB").save(partial, format="PNG")
        partial.rename(out / f"{scene_id}.png")


def fetch_scene_boxes(scene_ids: Iterable[str], out_dir: str | Path) -> None:
    """Download SCoT's per-scene box files, which define the scene image's coordinates."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in sorted(set(scene_ids)) if not (out / f"{s}_bbox.json").is_file()]
    if not wanted:
        return
    files = _list_drive_subfolder("scans_bbox")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    for scene_id in wanted:
        url = files.get(f"{scene_id}_bbox.json")
        if url is None:
            raise FileNotFoundError(f"{scene_id}_bbox.json is not in the SCoT Drive folder")
        file_id = re.search(r"/d/([^/?#]+)|id=([^&]+)", url)
        body = _read_drive_file(opener, file_id.group(1) or file_id.group(2))
        json.loads(body)  # A quota page must fail here, not later.
        partial = out / f"{scene_id}_bbox.json.partial"
        partial.write_bytes(body)
        partial.rename(out / f"{scene_id}_bbox.json")


def fit_scene_alignment(
    box_rows: list[list[float]],
    human_locs: Any,
    *,
    tolerance: float = 0.1,
    tries: int = 5000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Find the rotation and shift from object-list coordinates to scene image coordinates.

    SCoT's scene images use each scan's raw coordinates, while the object lists are
    axis-aligned. The two box lists describe the same objects in no shared order, so
    pairs are found by matching geometry (RANSAC), with height as a first filter.

    Args:
        box_rows: ``bbox`` rows of a scans_bbox file: center, half size, ids.
        human_locs: Human-annotated boxes of the same scene: center, full size.
        tolerance: Distance in meters for a pair to count as a match.
        tries: RANSAC attempts.
        seed: Random seed, so the result is repeatable.

    Returns:
        Alignment with ``rotation``, ``translation``, ``x_min``, ``y_min``, ``pairs``,
        and ``median_error``, or None when no two boxes can be paired.
    """
    import numpy as np

    boxes = np.array(box_rows, dtype=float)
    human = np.array(_as_rows(human_locs), dtype=float)
    known = np.where(boxes[:, 6] != -100)[0] if boxes.shape[1] > 6 else np.arange(len(boxes))
    candidates = [
        (i, j)
        for i in known
        for j in range(len(human))
        if abs(2 * boxes[i, 5] - human[j, 5]) < tolerance
    ]
    if len(candidates) < 2:
        return None

    def rigid(source: Any, target: Any) -> tuple[Any, Any]:
        source_mean, target_mean = source.mean(0), target.mean(0)
        u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
        rotation = vt.T @ np.diag([1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
        return rotation, target_mean - rotation @ source_mean

    def nearest(rotation: Any, translation: Any) -> tuple[Any, Any]:
        moved = human[:, :2] @ rotation.T + translation
        distance = np.linalg.norm(moved[:, None] - boxes[known][None, :, :2], axis=2)
        return distance.argmin(1), distance.min(1)

    rng = np.random.default_rng(seed)
    best_pairs, best = 0, None
    for _ in range(tries):
        first, second = rng.choice(len(candidates), 2, replace=False)
        (i1, j1), (i2, j2) = candidates[first], candidates[second]
        if i1 == i2 or j1 == j2:
            continue
        fit = rigid(human[[j1, j2], :2], boxes[[i1, i2], :2])
        pairs = int((nearest(*fit)[1] < tolerance).sum())
        if pairs > best_pairs:
            best_pairs, best = pairs, fit
    if best is None:
        return None

    index, distance = nearest(*best)
    matched = distance < tolerance
    rotation, translation = rigid(human[matched, :2], boxes[known[index[matched]], :2])
    moved = human[matched, :2] @ rotation.T + translation
    errors = np.linalg.norm(moved - boxes[known[index[matched]], :2], axis=1)
    return {
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "x_min": float((boxes[:, 0] - boxes[:, 3]).min()),
        "y_min": float((boxes[:, 1] - boxes[:, 4]).min()),
        "pairs": int(matched.sum()),
        "human_objects": len(human),
        "median_error": float(np.median(errors)),
    }


def alignment_is_usable(alignment: dict[str, Any] | None) -> bool:
    """Whether enough boxes agree to trust the fit."""
    return (
        alignment is not None
        and alignment["pairs"] >= MIN_ALIGNMENT_PAIRS
        and alignment["median_error"] <= MAX_ALIGNMENT_ERROR
    )


def to_pixels(points: Any, alignment: dict[str, Any], image_height: int) -> list[tuple[float, float]]:
    """Convert object-list x, y to pixels. Image rows run opposite to scene y."""
    import numpy as np

    moved = np.array(points, dtype=float) @ np.array(alignment["rotation"]).T + alignment["translation"]
    columns = (moved[:, 0] - alignment["x_min"]) / IMAGE_METERS_PER_PIXEL
    rows = image_height - (moved[:, 1] - alignment["y_min"]) / IMAGE_METERS_PER_PIXEL
    return list(zip(columns.tolist(), rows.tolist()))


def draw_object_map(
    image: Image.Image,
    objects: list[str],
    locs: Any,
    order: list[int],
    alignment: dict[str, Any],
) -> Image.Image:
    """Draw each object's box and shown id on a copy of the scene image.

    The shown ids match ``scene_text`` for the same ``order``. Ceilings, floors, and
    walls are left out because their boxes would cover the whole image; they stay in
    the text list. An id moves next to its object when its center is already taken.
    """
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font = ImageFont.load_default(size=16)
    width, height = marked.size
    rows = _as_rows(locs)
    placed: list[tuple[float, float, float, float]] = []
    for shown, original in enumerate(order):
        if objects[original] in UNDRAWN_OBJECTS:
            continue
        cx, cy, _, sx, sy, _ = rows[original]
        corners = [
            (cx - sx / 2, cy - sy / 2),
            (cx + sx / 2, cy - sy / 2),
            (cx + sx / 2, cy + sy / 2),
            (cx - sx / 2, cy + sy / 2),
        ]
        color = tuple(int(255 * v) for v in colorsys.hsv_to_rgb((shown * 0.618) % 1, 0.95, 1.0))
        draw.polygon(to_pixels(corners, alignment, height), outline=color, width=2)
        center = to_pixels([(cx, cy)], alignment, height)[0]
        label = f"{shown:03d}"
        spot = _free_label_spot(center, draw.textbbox((0, 0), label, font=font, anchor="mm"), placed, width, height)
        box = draw.textbbox(spot, label, font=font, anchor="mm")
        placed.append(box)
        if spot != center:
            draw.line([center, spot], fill=color, width=1)  # Ties a moved id to its object.
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text(spot, label, fill=color, font=font, anchor="mm")
    return marked


def _free_label_spot(
    center: tuple[float, float],
    size_box: tuple[float, float, float, float],
    placed: list[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> tuple[float, float]:
    """Nearest position around an object where its id overlaps no earlier id."""
    half_w = (size_box[2] - size_box[0]) / 2 + 1
    half_h = (size_box[3] - size_box[1]) / 2 + 1
    x, y = center
    offsets = [(0, 0)] + [
        (dx * half_w * 2 * ring, dy * half_h * 2 * ring)
        for ring in (1, 2, 3, 4)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
    ]
    for dx, dy in offsets:
        spot_x, spot_y = x + dx, y + dy
        inside = half_w <= spot_x <= width - half_w and half_h <= spot_y <= height - half_h
        clear = all(
            spot_x + half_w <= b[0] or spot_x - half_w >= b[2] or spot_y + half_h <= b[1] or spot_y - half_h >= b[3]
            for b in placed
        )
        if inside and clear:
            return spot_x, spot_y
    return x, y  # shortcut: a very crowded spot may still overlap; rare, so accepted for now.


def extract_frames(video_path: str | Path, out_dir: str | Path, num_frames: int = 32) -> None:
    """Save evenly spaced JPEG frames, so model input does not depend on a video decoder.

    Raises:
        ValueError: The video cannot be read.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Cannot read video: {video_path}")
    out = Path(out_dir)
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    count = min(num_frames, total)
    wanted = {round(i * (total - 1) / max(count - 1, 1)) for i in range(count)}
    saved = 0
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            cv2.imwrite(str(partial / f"{saved:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
    capture.release()
    if saved == 0:
        raise ValueError(f"No frames read from: {video_path}")
    partial.rename(out)


def fetch_scene_frames(scene_ids: Iterable[str], out_dir: str | Path, num_frames: int = 32) -> None:
    """Extract frames for scenes from VSI-Bench's ScanNet videos, skipping finished scenes."""
    from huggingface_hub import hf_hub_download

    out = Path(out_dir)
    wanted = [s for s in sorted(set(scene_ids)) if not (out / s).is_dir()]
    if not wanted:
        return
    out.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(VIDEO_REPO, "scannet.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as videos:
        for scene_id in wanted:
            member = f"scannet/{scene_id}.mp4"
            with videos.open(member) as source, tempfile.NamedTemporaryFile(suffix=".mp4", dir=out) as temp:
                shutil.copyfileobj(source, temp)
                temp.flush()
                extract_frames(temp.name, out / scene_id, num_frames)


def _list_drive_subfolder(name: str) -> dict[str, str]:
    folders = _list_drive_folder(IMAGE_FOLDER_ID)
    return _list_drive_folder(re.search(r"/folders/([^/?#]+)", folders[name]).group(1))


def _list_drive_folder(folder_id: str) -> dict[str, str]:
    request = urllib.request.Request(
        f"https://drive.google.com/embeddedfolderview?id={folder_id}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8", "replace")
    pattern = r'<a href="([^"]+)"[^>]*>.*?flip-entry-title">(.*?)</div>'
    return {html.unescape(m.group(2)): html.unescape(m.group(1)) for m in re.finditer(pattern, page, re.S)}


def _read_drive_file(opener: Any, file_id: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with opener.open(urllib.request.Request(url, headers=headers), timeout=300) as response:
        body = response.read()
    form = re.search(rb'action="([^"]+)"', body) if b"<html" in body[:2000].lower() else None
    if form is None:
        return body
    # Large files return a confirmation form first.
    fields = re.findall(rb'name="([^"]+)" value="([^"]*)"', body)
    query = "&".join(f"{k.decode()}={v.decode()}" for k, v in fields)
    target = f"{html.unescape(form.group(1).decode())}?{query}"
    with opener.open(urllib.request.Request(target, headers=headers), timeout=300) as response:
        return response.read()


def parse_object_id(answer: str, object_count: int) -> int:
    """Return the first valid object id, or 0 when there is none, as SCoT's evaluator does."""
    for match in _OBJECT_ID.finditer(answer):
        object_id = int(match.group(1))
        if object_id < object_count:
            return object_id
    return 0


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """IoU of two axis-aligned boxes given as center x, y, z and size x, y, z."""
    a, b = list(box_a), list(box_b)
    overlap = 1.0
    for axis in range(3):
        low = max(a[axis] - a[axis + 3] / 2, b[axis] - b[axis + 3] / 2)
        high = min(a[axis] + a[axis + 3] / 2, b[axis] + b[axis + 3] / 2)
        overlap *= max(0.0, high - low)
    union = a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - overlap
    return overlap / union if union > 0 else 0.0


def implicit_hit(
    answer: str,
    mask3d_locs: Any,
    target_loc: Any,
    threshold: float = 0.25,
    object_order: list[int] | None = None,
) -> bool:
    """SCoT's Acc@threshold: the chosen Mask3D box overlaps the annotated target enough.

    object_order maps the id shown to the model back to the original object.
    """
    rows = _as_rows(mask3d_locs)
    shown = parse_object_id(answer, len(rows))
    original = object_order[shown] if object_order is not None else shown
    return box_iou(rows[original], _as_row(target_loc)) >= threshold


def random_hit_rate(mask3d_locs: Any, target_loc: Any, threshold: float = 0.25) -> float:
    """Chance that a uniformly random Mask3D object would pass ``implicit_hit``."""
    rows = _as_rows(mask3d_locs)
    target = _as_row(target_loc)
    return sum(box_iou(row, target) >= threshold for row in rows) / len(rows)


def true_distance(prompt: str) -> float | None:
    """Euclidean distance between the two coordinates in a Relationship question."""
    found = _COORDINATES.findall(prompt)
    if len(found) < 2:
        return None
    return math.dist([float(v) for v in found[0]], [float(v) for v in found[1]])


def distance_error(answer: str, prompt: str) -> float | None:
    """Absolute error of the stated distance against the coordinates in the question.

    shortcut: answers mix sizes and distances, so the closest number in metres is used;
    this can overcount correct answers, so upgrade to a fixed answer format if it matters.
    """
    truth = true_distance(prompt)
    stated = [float(value) for value in _METERS.findall(answer)]
    if truth is None or not stated:
        return None
    return min(abs(value - truth) for value in stated)


# Copied from SCoT utils/helper.py (MIT), which credits LEO: embodied-generalist.
def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub("[^a-zA-Z0-9,'\\s\\-:]+", '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def _clean_prediction(text: str) -> str:
    """Prediction cleanup from SCoT's calc_scot_score."""
    if len(text) > 1:
        if text[-1] == ".":
            text = text[:-1]
        text = text[0].lower() + text[1:]
    return clean_answer(text)


def caption_scores(preds: dict[str, str], refs: dict[str, list[str]]) -> dict[str, float]:
    """SCoT's text metrics: same cleanup, tokenizer, and scorers. METEOR needs Java."""
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tokenizer = PTBTokenizer()
    # METEOR cannot score an empty string, so empty answers become a placeholder word.
    res = tokenizer.tokenize(
        {k: [{"caption": _clean_prediction(v.strip()) or "none"}] for k, v in preds.items()}
    )
    gts = tokenizer.tokenize(
        {k: [{"caption": clean_answer(c).replace("\n", " ").strip()} for c in refs[k]] for k in preds}
    )
    return {
        "rouge_l": float(Rouge().compute_score(gts, res)[0]),
        "meteor": float(Meteor().compute_score(gts, res)[0]),
    }


def is_correct(
    answer: str,
    meta: dict[str, Any],
    mask3d: dict[str, dict[str, Any]],
    human: dict[str, dict[str, Any]],
    *,
    distance_tolerance: float = 0.2,
) -> bool | None:
    """Program-graded correctness. None for tasks that need a text metric or a judge."""
    if meta["task"] == "implicit_detection":
        target = _as_rows(human[meta["scene_id"]]["locs"])[int(meta["obj_id"])]
        locs = mask3d[meta["scene_id"]]["locs"]
        return implicit_hit(answer, locs, target, 0.25, meta.get("object_order"))
    if meta["task"] == "relationship_analysis":
        error = distance_error(answer, meta["prompt"])
        return error is not None and error <= distance_tolerance
    return None


def answer_text(row: dict[str, Any], arm: str) -> str:
    """The answer to score. Instruct has no think tag, so all of its output counts."""
    return row["raw_output"] if arm.startswith("instruct") else row["final_answer"]


def summarize_arm(
    rows: list[dict[str, Any]],
    arm: str,
    mask3d: dict[str, dict[str, Any]],
    human: dict[str, dict[str, Any]],
    *,
    distance_tolerance: float = 0.2,
) -> list[dict[str, Any]]:
    """Per-task table of reasoning length, speed, memory, and program-graded metrics."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["metadata"]["task"], []).append(row)

    table = []
    for task in sorted(by_task):
        group = by_task[task]
        reasoning = sorted(row["reasoning_token_count"] for row in group)
        generated = sum(row["total_generated_token_count"] for row in group)
        seconds = sum(row["latency_seconds"] for row in group)
        vram = [row["metadata"].get("peak_vram_gb") for row in group]
        vram = [value for value in vram if value is not None]
        summary: dict[str, Any] = {
            "task": task,
            "arm": arm,
            "n": len(group),
            "reasoning_tokens_median": 0 if arm.startswith("instruct") else statistics.median(reasoning),
            "reasoning_tokens_p90": (
                0 if arm.startswith("instruct") else reasoning[int(0.9 * (len(reasoning) - 1))]
            ),
            "generated_tokens_median": statistics.median(
                row["total_generated_token_count"] for row in group
            ),
            "truncated_rate": (
                0.0
                if arm.startswith("instruct")
                else sum(row["parse_status"] == "missing_think_end" for row in group) / len(group)
            ),
            "latency_seconds_median": statistics.median(row["latency_seconds"] for row in group),
            "tokens_per_second": generated / seconds if seconds else 0.0,
            "peak_vram_gb": max(vram) if vram else None,
        }
        if task == "implicit_detection":
            hits = []
            hits_050 = []
            chance = []
            for row in group:
                meta = row["metadata"]
                locs = mask3d[meta["scene_id"]]["locs"]
                target = _as_rows(human[meta["scene_id"]]["locs"])[int(meta["obj_id"])]
                order = meta.get("object_order")
                hits.append(implicit_hit(answer_text(row, arm), locs, target, 0.25, order))
                hits_050.append(implicit_hit(answer_text(row, arm), locs, target, 0.5, order))
                chance.append(random_hit_rate(locs, target))
            summary["acc_at_025"] = sum(hits) / len(hits)
            summary["acc_at_050"] = sum(hits_050) / len(hits_050)
            summary["acc_at_025_random"] = sum(chance) / len(chance)
        if task == "relationship_analysis":
            errors = [
                distance_error(answer_text(row, arm), row["metadata"]["prompt"]) for row in group
            ]
            summary["distance_ok_rate"] = sum(
                error is not None and error <= distance_tolerance for error in errors
            ) / len(errors)
        table.append(summary)
    return table


def _as_rows(locs: Any) -> list[list[float]]:
    return locs.tolist() if hasattr(locs, "tolist") else [list(row) for row in locs]


def _as_row(loc: Any) -> list[float]:
    return loc.tolist() if hasattr(loc, "tolist") else list(loc)
