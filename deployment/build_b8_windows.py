from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO


T = 9
MAX_PLAYERS = 12

CROP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@dataclass
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class FrameObservation:
    frame_idx: int
    frame_bgr: np.ndarray
    detections: list[Detection]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build B8-compatible 12 x 9 temporal player windows from a raw video "
            "using the trained YOLO detector and ByteTrack."
        )
    )
    p.add_argument("--weights", required=True, help="YOLO player detector best.pt")
    p.add_argument("--source", required=True, help="Raw volleyball MP4/AVI")
    p.add_argument(
        "--segments",
        required=True,
        help="gameplay_segments.csv produced by track_gameplay.py",
    )
    p.add_argument(
        "--tracker",
        default="players_detection/bytetrack_volleyball.yaml",
    )
    p.add_argument(
        "--output-dir",
        default="./runs/b8_windows",
    )
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--window-stride",
        type=int,
        default=9,
        help="Start a new 9-frame B8 window every N video frames.",
    )
    p.add_argument(
        "--save-montages",
        type=int,
        default=8,
        help="Save this many anchor-frame 12-slot montages for visual inspection.",
    )
    p.add_argument(
        "--save-tensors",
        type=int,
        default=0,
        help=(
            "Save this many raw .pt tensors. Each tensor is large (~65 MB), "
            "so default is 0."
        ),
    )
    return p.parse_args()


def reset_tracker(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()


def read_segments(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "segment_id": int(row["segment_id"]),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                }
            )
    return rows


def clamp_box(
    bbox: tuple[float, float, float, float],
    w: int,
    h: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def select_anchor_players(
    detections: list[Detection],
) -> list[Detection]:
    if len(detections) > MAX_PLAYERS:
        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True,
        )[:MAX_PLAYERS]
    return sorted(detections, key=lambda d: d.bbox[0])


def crop_to_tensor(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> torch.Tensor | None:
    h, w = frame_bgr.shape[:2]
    box = clamp_box(bbox, w, h)
    if box is None:
        return None

    x1, y1, x2, y2 = box
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    crop = image.crop((x1, y1, x2, y2))
    return CROP_TRANSFORM(crop)


def build_b8_tensor(
    observations: list[FrameObservation],
) -> tuple[torch.Tensor, dict[int, int], dict]:
    if len(observations) != T:
        raise ValueError(f"Expected exactly {T} observations, got {len(observations)}")

    anchor_players = select_anchor_players(observations[0].detections)
    player_id_to_slot = {
        det.track_id: slot for slot, det in enumerate(anchor_players)
    }

    zero_crop = torch.zeros(3, 224, 224)
    temporal_frames = []
    filled_cells = 0

    for obs in observations:
        crops = [zero_crop.clone() for _ in range(MAX_PLAYERS)]

        for det in obs.detections:
            pid = det.track_id

            if pid not in player_id_to_slot:
                if len(player_id_to_slot) < MAX_PLAYERS:
                    player_id_to_slot[pid] = len(player_id_to_slot)
                else:
                    continue

            slot = player_id_to_slot[pid]
            if slot >= MAX_PLAYERS:
                continue

            crop = crop_to_tensor(obs.frame_bgr, det.bbox)
            if crop is not None:
                crops[slot] = crop
                filled_cells += 1

        temporal_frames.append(torch.stack(crops, dim=0))

    tensor = torch.stack(temporal_frames, dim=0).permute(1, 0, 2, 3, 4)
    coverage = filled_cells / float(MAX_PLAYERS * T)

    metadata = {
        "anchor_detections": len(observations[0].detections),
        "anchor_selected": len(anchor_players),
        "allocated_track_ids": len(player_id_to_slot),
        "filled_cells": filled_cells,
        "coverage": coverage,
    }

    anchor_ids = {slot: det.track_id for slot, det in enumerate(anchor_players)}
    return tensor, anchor_ids, metadata


def make_anchor_montage(
    obs: FrameObservation,
    anchor_ids: dict[int, int],
    out_path: Path,
) -> None:
    track_to_det = {d.track_id: d for d in obs.detections}
    tile_w, tile_h = 180, 150
    cols, rows = 6, 2
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    frame_rgb = cv2.cvtColor(obs.frame_bgr, cv2.COLOR_BGR2RGB)
    frame_img = Image.fromarray(frame_rgb)
    fw, fh = frame_img.size

    for slot in range(MAX_PLAYERS):
        x = (slot % cols) * tile_w
        y = (slot // cols) * tile_h

        if slot in anchor_ids and anchor_ids[slot] in track_to_det:
            det = track_to_det[anchor_ids[slot]]
            box = clamp_box(det.bbox, fw, fh)
            if box is not None:
                crop = frame_img.crop(box)
                crop.thumbnail((tile_w - 8, tile_h - 30))

                bg = Image.new("RGB", (tile_w - 8, tile_h - 30), (0, 0, 0))
                px = (bg.width - crop.width) // 2
                py = (bg.height - crop.height) // 2
                bg.paste(crop, (px, py))
                canvas.paste(bg, (x + 4, y + 26))

                draw.text(
                    (x + 5, y + 5),
                    f"slot {slot} | track {det.track_id} | {det.confidence:.2f}",
                    fill=(255, 255, 255),
                )
                continue

        draw.text((x + 5, y + 5), f"slot {slot} | EMPTY", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def track_frame(
    model: YOLO,
    frame: np.ndarray,
    args: argparse.Namespace,
) -> list[Detection]:
    kwargs = dict(
        source=frame,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=[0],
        verbose=False,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    result = model.track(**kwargs)[0]

    if (
        result.boxes is None
        or not result.boxes.is_track
        or result.boxes.id is None
    ):
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()

    return [
        Detection(
            track_id=int(track_id),
            bbox=tuple(map(float, xyxy)),
            confidence=float(conf),
        )
        for xyxy, conf, track_id in zip(boxes, confs, ids)
    ]


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    segments_path = Path(args.segments)
    out_dir = Path(args.output_dir)

    for p in (weights, source, segments_path):
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = out_dir / "anchor_montages"
    tensor_dir = out_dir / "sample_tensors"

    model = YOLO(str(weights))
    segments = read_segments(segments_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_rows = []
    montage_saved = 0
    tensors_saved = 0
    global_window_id = 0

    print(f"Source: {source}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"B8 temporal length T: {T}")
    print(f"Window stride: {args.window_stride}")
    print(f"Gameplay segments: {len(segments)}")
    print()

    for seg in segments:
        segment_id = seg["segment_id"]
        start = seg["start_frame"]
        end = seg["end_frame"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        reset_tracker(model)

        observations: deque[FrameObservation] = deque(maxlen=T)
        segment_window_counter = 0
        frame_idx = start

        while frame_idx <= end:
            ok, frame = cap.read()
            if not ok:
                break

            detections = track_frame(model, frame, args)
            observations.append(
                FrameObservation(
                    frame_idx=frame_idx,
                    frame_bgr=frame.copy(),
                    detections=detections,
                )
            )

            if len(observations) == T:
                window_start = observations[0].frame_idx

                if (window_start - start) % args.window_stride == 0:
                    tensor, anchor_ids, meta = build_b8_tensor(list(observations))

                    if tuple(tensor.shape) != (12, 9, 3, 224, 224):
                        raise RuntimeError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

                    global_window_id += 1
                    segment_window_counter += 1

                    window_rows.append(
                        {
                            "window_id": global_window_id,
                            "segment_id": segment_id,
                            "start_frame": observations[0].frame_idx,
                            "end_frame": observations[-1].frame_idx,
                            "start_time_sec": round(observations[0].frame_idx / fps, 3),
                            "end_time_sec": round(observations[-1].frame_idx / fps, 3),
                            "anchor_detections": meta["anchor_detections"],
                            "anchor_selected": meta["anchor_selected"],
                            "allocated_track_ids": meta["allocated_track_ids"],
                            "filled_cells": meta["filled_cells"],
                            "coverage": round(meta["coverage"], 4),
                            "tensor_shape": "12x9x3x224x224",
                            "anchor_track_ids": ",".join(
                                str(anchor_ids[s]) if s in anchor_ids else ""
                                for s in range(MAX_PLAYERS)
                            ),
                        }
                    )

                    if montage_saved < args.save_montages:
                        montage_path = (
                            montage_dir
                            / f"window_{global_window_id:04d}_segment_{segment_id}.jpg"
                        )
                        make_anchor_montage(
                            observations[0],
                            anchor_ids,
                            montage_path,
                        )
                        montage_saved += 1

                    if tensors_saved < args.save_tensors:
                        tensor_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            tensor,
                            tensor_dir / f"window_{global_window_id:04d}.pt",
                        )
                        tensors_saved += 1

            frame_idx += 1

        print(
            f"Segment {segment_id}: {start}-{end} | "
            f"B8 windows={segment_window_counter}"
        )

    cap.release()

    csv_path = out_dir / "b8_windows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "window_id",
            "segment_id",
            "start_frame",
            "end_frame",
            "start_time_sec",
            "end_time_sec",
            "anchor_detections",
            "anchor_selected",
            "allocated_track_ids",
            "filled_cells",
            "coverage",
            "tensor_shape",
            "anchor_track_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(window_rows)

    if window_rows:
        coverages = np.asarray([r["coverage"] for r in window_rows], dtype=float)
        selected = np.asarray([r["anchor_selected"] for r in window_rows], dtype=float)

        print("\nB8 window summary")
        print(f"Total windows: {len(window_rows)}")
        print(f"Mean slot/frame coverage: {coverages.mean():.3f}")
        print(f"Median slot/frame coverage: {np.median(coverages):.3f}")
        print(f"Windows with >= 10 anchor players: {(selected >= 10).mean():.1%}")
        print(f"Windows with 12 anchor players: {(selected == 12).mean():.1%}")
        print(f"Metadata CSV: {csv_path.resolve()}")
        print(f"Anchor montages: {montage_dir.resolve()}")
        if args.save_tensors:
            print(f"Sample tensors: {tensor_dir.resolve()}")
    else:
        print("No valid 9-frame B8 windows were generated.")


if __name__ == "__main__":
    main()
