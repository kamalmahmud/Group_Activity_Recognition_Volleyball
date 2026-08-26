from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image

# Import these before unpickling so BoxInfo/classes used by the existing project
# are available under their normal module paths.
from data.boxinfo import BoxInfo  # noqa: F401
from data.constants import SPLITS


def yolo_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Clamp an XYXY box and convert it to normalized YOLO XYWH."""
    x1 = max(0.0, min(float(x1), float(image_width)))
    y1 = max(0.0, min(float(y1), float(image_height)))
    x2 = max(0.0, min(float(x2), float(image_width)))
    y2 = max(0.0, min(float(y2), float(image_height)))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / image_width
    yc = ((y1 + y2) / 2.0) / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height

    if w <= 0.0 or h <= 0.0:
        return None

    return xc, yc, w, h


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    """Create an image entry using symlink/hardlink/copy with safe fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass

    if mode in {"symlink", "hardlink"}:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def select_frames(frame_ids: Iterable[int], clip_id: str, stride: int) -> list[int]:
    """
    Sample frames to control detector-training size while always keeping
    the annotated/key frame when it is present.
    """
    frames = sorted(int(f) for f in frame_ids)
    if stride <= 1:
        return frames

    selected = set(frames[::stride])

    try:
        key_frame = int(clip_id)
        if key_frame in frames:
            selected.add(key_frame)
    except ValueError:
        pass

    return sorted(selected)


def write_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "data.yaml"
    content = f"""path: {output_root.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: player
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def build_split(
    annotations: dict,
    videos_root: Path,
    output_root: Path,
    split_name: str,
    video_ids: list[str],
    frame_stride: int,
    link_mode: str,
    include_lost: bool,
    include_generated: bool,
) -> dict[str, int]:
    images_out = output_root / "images" / split_name
    labels_out = output_root / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "boxes": 0,
        "skipped_missing_images": 0,
        "skipped_invalid_boxes": 0,
        "skipped_lost_boxes": 0,
        "skipped_generated_boxes": 0,
        "empty_images": 0,
    }

    for video_id in video_ids:
        video_clips = annotations.get(str(video_id), {})

        for clip_id, clip_data in video_clips.items():
            frame_boxes = clip_data.get("frame_boxes_dct", {})
            frame_ids = select_frames(frame_boxes.keys(), str(clip_id), frame_stride)

            for frame_id in frame_ids:
                src_image = (
                    videos_root
                    / str(video_id)
                    / str(clip_id)
                    / f"{frame_id}.jpg"
                )

                if not src_image.exists():
                    stats["skipped_missing_images"] += 1
                    continue

                # Unique name prevents collisions between videos/clips.
                stem = f"v{video_id}_c{clip_id}_f{frame_id}"
                dst_image = images_out / f"{stem}.jpg"
                dst_label = labels_out / f"{stem}.txt"

                with Image.open(src_image) as img:
                    width, height = img.size

                labels = []
                for box_info in frame_boxes.get(frame_id, []):
                    if not include_lost and int(getattr(box_info, "lost", 0)) != 0:
                        stats["skipped_lost_boxes"] += 1
                        continue

                    if not include_generated and int(getattr(box_info, "generated", 0)) != 0:
                        stats["skipped_generated_boxes"] += 1
                        continue

                    x1, y1, x2, y2 = box_info.box
                    converted = yolo_box(x1, y1, x2, y2, width, height)
                    if converted is None:
                        stats["skipped_invalid_boxes"] += 1
                        continue

                    xc, yc, bw, bh = converted
                    labels.append(
                        f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    )

                materialize_image(src_image, dst_image, link_mode)
                dst_label.write_text("\n".join(labels), encoding="utf-8")

                stats["images"] += 1
                stats["boxes"] += len(labels)
                if not labels:
                    stats["empty_images"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a YOLO player-detection dataset from annot_all.pkl."
    )
    parser.add_argument("--pkl", required=True, help="Path to annot_all.pkl")
    parser.add_argument(
        "--videos",
        required=True,
        help="Path to the Volleyball Dataset videos/ directory",
    )
    parser.add_argument(
        "--output",
        default="./yolo_volleyball",
        help="YOLO dataset output directory",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help=(
            "Use every Nth tracked frame from each clip. "
            "The clip/key frame is always included. Use 1 for all frames."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize images in the YOLO dataset.",
    )
    parser.add_argument(
        "--include-lost",
        action="store_true",
        help="Include boxes whose tracking annotation marks them as lost.",
    )
    parser.add_argument(
        "--exclude-generated",
        action="store_true",
        help="Exclude boxes marked as generated/interpolated.",
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    videos_root = Path(args.videos)
    output_root = Path(args.output)

    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    if not videos_root.exists():
        raise FileNotFoundError(videos_root)

    with pkl_path.open("rb") as f:
        annotations = pickle.load(f)

    print(f"Loaded annotations from: {pkl_path}")
    print(f"Videos root: {videos_root}")
    print(f"Output root: {output_root}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Image mode: {args.link_mode}")
    print()

    all_stats = {}
    for split_name in ("train", "val", "test"):
        stats = build_split(
            annotations=annotations,
            videos_root=videos_root,
            output_root=output_root,
            split_name=split_name,
            video_ids=SPLITS[split_name],
            frame_stride=args.frame_stride,
            link_mode=args.link_mode,
            include_lost=args.include_lost,
            include_generated=not args.exclude_generated,
        )
        all_stats[split_name] = stats
        print(f"{split_name.upper()}: {stats}")

    yaml_path = write_yaml(output_root)
    print(f"\nCreated YOLO config: {yaml_path}")

    total_images = sum(x["images"] for x in all_stats.values())
    total_boxes = sum(x["boxes"] for x in all_stats.values())
    print(f"Total images: {total_images}")
    print(f"Total player boxes: {total_boxes}")


if __name__ == "__main__":
    main()
