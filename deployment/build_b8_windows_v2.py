from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
class Det:
    frame_pos: int
    track_id: int
    bbox: np.ndarray
    conf: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Window-local YOLO+ByteTrack builder for B8 [12,9,3,224,224] input."
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument(
        "--tracker",
        default="players_detection/bytetrack_b8_window.yaml",
    )
    p.add_argument("--output-dir", default="./runs/b8_windows_v2")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--device", default=None)
    p.add_argument("--window-stride", type=int, default=9)
    p.add_argument("--min-track-frames", type=int, default=4)
    p.add_argument("--min-observed-coverage", type=float, default=0.80)
    p.add_argument("--max-edge-fill-gap", type=int, default=2)
    p.add_argument("--save-montages", type=int, default=12)
    p.add_argument("--save-tensors", type=int, default=0)
    return p.parse_args()


def read_segments(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                (
                    int(row["segment_id"]),
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                )
            )
    return out


def reset_tracker(model: YOLO):
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            fn = getattr(tracker, "reset", None)
            if callable(fn):
                fn()


def run_track(model, frame, args):
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
    if result.boxes is None or not result.boxes.is_track or result.boxes.id is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()

    return [
        (int(tid), np.asarray(box, dtype=np.float32), float(conf))
        for tid, box, conf in zip(ids, boxes, confs)
    ]


def interpolate_boxes(track_boxes, max_edge_fill_gap):
    """
    track_boxes: dict frame_pos -> xyxy
    Returns list[9] of box or None, plus observed/interpolated counts.
    """
    result = [None] * T
    observed_positions = sorted(track_boxes)

    if not observed_positions:
        return result, 0, 0

    for pos, box in track_boxes.items():
        result[pos] = box.copy()

    # Linear interpolation only between genuine detections.
    for left, right in zip(observed_positions[:-1], observed_positions[1:]):
        if right - left <= 1:
            continue
        b0 = track_boxes[left]
        b1 = track_boxes[right]
        for pos in range(left + 1, right):
            alpha = (pos - left) / float(right - left)
            result[pos] = (1.0 - alpha) * b0 + alpha * b1

    # Short carry at edges only.
    first = observed_positions[0]
    for pos in range(max(0, first - max_edge_fill_gap), first):
        result[pos] = track_boxes[first].copy()

    last = observed_positions[-1]
    for pos in range(last + 1, min(T, last + max_edge_fill_gap + 1)):
        result[pos] = track_boxes[last].copy()

    observed = len(observed_positions)
    effective = sum(x is not None for x in result)
    return result, observed, effective


def clamp_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_tensor(frame_bgr, box):
    h, w = frame_bgr.shape[:2]
    b = clamp_box(box, w, h)
    if b is None:
        return None
    x1, y1, x2, y2 = b
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(rgb).crop((x1, y1, x2, y2))
    return CROP_TRANSFORM(crop)


def estimate_anchor_x(boxes):
    # Prefer true frame-0 box; otherwise earliest available estimated box.
    if boxes[0] is not None:
        return float(boxes[0][0])
    for box in boxes:
        if box is not None:
            return float(box[0])
    return float("inf")


def choose_tracks(track_data, min_track_frames):
    scored = []
    for tid, item in track_data.items():
        observed = len(item["boxes"])
        if observed < min_track_frames:
            continue
        mean_conf = float(np.mean(item["confs"])) if item["confs"] else 0.0
        scored.append((tid, observed, mean_conf))

    # Presence is more important than confidence for B8.
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in scored[:MAX_PLAYERS]]


def make_montage(frames, selected, filled_boxes, track_data, path):
    tile_w, tile_h = 180, 150
    canvas = Image.new("RGB", (6 * tile_w, 2 * tile_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    anchor_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    anchor_img = Image.fromarray(anchor_rgb)
    fw, fh = anchor_img.size

    for slot in range(MAX_PLAYERS):
        x = (slot % 6) * tile_w
        y = (slot // 6) * tile_h
        if slot >= len(selected):
            draw.text((x + 5, y + 5), f"slot {slot} EMPTY", fill=(255, 255, 255))
            continue

        tid = selected[slot]
        box = filled_boxes[tid][0]
        observed0 = 0 in track_data[tid]["boxes"]

        if box is None:
            draw.text((x + 5, y + 5), f"slot {slot} T{tid} NO BOX", fill=(255, 255, 255))
            continue

        b = clamp_box(box, fw, fh)
        if b is None:
            continue
        crop = anchor_img.crop(b)
        crop.thumbnail((tile_w - 8, tile_h - 32))
        bg = Image.new("RGB", (tile_w - 8, tile_h - 32), (0, 0, 0))
        bg.paste(crop, ((bg.width-crop.width)//2, (bg.height-crop.height)//2))
        canvas.paste(bg, (x + 4, y + 28))
        src = "OBS" if observed0 else "FILL"
        draw.text((x + 5, y + 5), f"slot {slot} T{tid} {src}", fill=(255,255,255))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)
    segments_path = Path(args.segments)
    out_dir = Path(args.output_dir)

    for p in [weights, source, segments_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = out_dir / "anchor_montages"
    tensor_dir = out_dir / "sample_tensors"

    segments = read_segments(segments_path)
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

    rows = []
    window_id = 0
    montage_saved = 0
    tensor_saved = 0

    for segment_id, seg_start, seg_end in segments:
        starts = list(range(seg_start, seg_end - T + 2, args.window_stride))

        for start in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            reset_tracker(model)

            frames = []
            track_data = defaultdict(lambda: {"boxes": {}, "confs": []})
            raw_counts = []

            ok_window = True
            for pos in range(T):
                ok, frame = cap.read()
                if not ok:
                    ok_window = False
                    break
                frames.append(frame.copy())

                dets = run_track(model, frame, args)
                raw_counts.append(len(dets))
                for tid, box, conf in dets:
                    track_data[tid]["boxes"][pos] = box
                    track_data[tid]["confs"].append(conf)

            if not ok_window or len(frames) != T:
                continue

            chosen_ids = choose_tracks(track_data, args.min_track_frames)

            filled_boxes = {}
            observed_cells = 0
            effective_cells = 0

            for tid in chosen_ids:
                seq, observed, effective = interpolate_boxes(
                    track_data[tid]["boxes"],
                    args.max_edge_fill_gap,
                )
                filled_boxes[tid] = seq
                observed_cells += observed
                effective_cells += effective

            # Sort the selected tracks left-to-right by their anchor-frame estimate.
            chosen_ids.sort(key=lambda tid: estimate_anchor_x(filled_boxes[tid]))

            observed_coverage = observed_cells / float(MAX_PLAYERS * T)
            effective_coverage = effective_cells / float(MAX_PLAYERS * T)

            accepted = (
                len(chosen_ids) == MAX_PLAYERS
                and observed_coverage >= args.min_observed_coverage
            )

            window_id += 1

            tensor = None
            if accepted:
                zero = torch.zeros(3, 224, 224)
                slots = []
                for tid in chosen_ids:
                    temporal = []
                    for pos in range(T):
                        box = filled_boxes[tid][pos]
                        crop = crop_tensor(frames[pos], box) if box is not None else None
                        temporal.append(crop if crop is not None else zero.clone())
                    slots.append(torch.stack(temporal, dim=0))
                tensor = torch.stack(slots, dim=0)

                if tuple(tensor.shape) != (12, 9, 3, 224, 224):
                    raise RuntimeError(tuple(tensor.shape))

            rows.append(
                {
                    "window_id": window_id,
                    "segment_id": segment_id,
                    "start_frame": start,
                    "key_frame": start + 5,
                    "end_frame": start + 8,
                    "key_time_sec": round((start + 5) / fps, 3),
                    "mean_raw_detections": round(float(np.mean(raw_counts)), 3),
                    "min_raw_detections": int(min(raw_counts)),
                    "max_raw_detections": int(max(raw_counts)),
                    "candidate_tracks": len(track_data),
                    "selected_tracks": len(chosen_ids),
                    "observed_cells": observed_cells,
                    "effective_cells": effective_cells,
                    "observed_coverage": round(observed_coverage, 4),
                    "effective_coverage": round(effective_coverage, 4),
                    "accepted": int(accepted),
                    "track_ids_left_to_right": ",".join(map(str, chosen_ids)),
                }
            )

            if montage_saved < args.save_montages and len(chosen_ids) > 0:
                make_montage(
                    frames,
                    chosen_ids,
                    filled_boxes,
                    track_data,
                    montage_dir / f"window_{window_id:04d}_accepted_{int(accepted)}.jpg",
                )
                montage_saved += 1

            if accepted and tensor is not None and tensor_saved < args.save_tensors:
                tensor_dir.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, tensor_dir / f"window_{window_id:04d}.pt")
                tensor_saved += 1

        print(f"Segment {segment_id}: tested {len(starts)} windows")

    cap.release()

    csv_path = out_dir / "b8_windows_v2.csv"
    fields = [
        "window_id","segment_id","start_frame","key_frame","end_frame","key_time_sec",
        "mean_raw_detections","min_raw_detections","max_raw_detections",
        "candidate_tracks","selected_tracks","observed_cells","effective_cells",
        "observed_coverage","effective_coverage","accepted","track_ids_left_to_right"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if rows:
        obs = np.asarray([r["observed_coverage"] for r in rows], dtype=float)
        eff = np.asarray([r["effective_coverage"] for r in rows], dtype=float)
        raw = np.asarray([r["mean_raw_detections"] for r in rows], dtype=float)
        sel = np.asarray([r["selected_tracks"] for r in rows], dtype=int)
        acc = np.asarray([r["accepted"] for r in rows], dtype=int)

        print("\n===== B8 WINDOW V2 SUMMARY =====")
        print(f"Windows tested: {len(rows)}")
        print(f"Mean raw detections/frame: {raw.mean():.3f}")
        print(f"Median raw detections/frame: {np.median(raw):.3f}")
        print(f"Mean observed coverage: {obs.mean():.3f}")
        print(f"Median observed coverage: {np.median(obs):.3f}")
        print(f"Mean effective coverage after short interpolation: {eff.mean():.3f}")
        print(f"Windows with 12 selected trajectories: {(sel == 12).mean():.1%}")
        print(f"Accepted windows: {acc.mean():.1%} ({acc.sum()}/{len(acc)})")

        if acc.sum():
            mask = acc == 1
            print(f"Accepted mean observed coverage: {obs[mask].mean():.3f}")
            print(f"Accepted mean effective coverage: {eff[mask].mean():.3f}")

        print(f"CSV: {csv_path.resolve()}")
        print(f"Montages: {montage_dir.resolve()}")


if __name__ == "__main__":
    main()
