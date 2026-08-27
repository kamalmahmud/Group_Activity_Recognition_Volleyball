from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
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
    bbox: np.ndarray
    conf: float


@dataclass
class Tracklet:
    tid: int
    boxes: dict[int, np.ndarray] = field(default_factory=dict)
    confs: dict[int, float] = field(default_factory=dict)

    @property
    def observed(self) -> int:
        return len(self.boxes)

    @property
    def first_pos(self) -> int:
        return min(self.boxes) if self.boxes else 999

    @property
    def last_pos(self) -> int:
        return max(self.boxes) if self.boxes else -1

    @property
    def avg_conf(self) -> float:
        return float(np.mean(list(self.confs.values()))) if self.confs else 0.0

    def last_box_before(self, pos: int):
        candidates = [p for p in self.boxes if p < pos]
        if not candidates:
            return None, None
        p = max(candidates)
        return p, self.boxes[p]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build B8 windows using raw YOLO detections and short-horizon "
            "Hungarian association instead of ByteTrack."
        )
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--output-dir", default="./runs/b8_hungarian_v3")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.03)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--device", default=None)
    p.add_argument("--window-stride", type=int, default=9)

    # Matching / gating
    p.add_argument("--iou-weight", type=float, default=0.65)
    p.add_argument("--center-weight", type=float, default=0.35)
    p.add_argument("--max-center-dist", type=float, default=0.16)
    p.add_argument("--min-iou-gate", type=float, default=0.01)
    p.add_argument("--max-gap", type=int, default=2)

    # Window acceptance
    p.add_argument("--min-track-frames", type=int, default=6)
    p.add_argument("--min-observed-coverage", type=float, default=0.85)
    p.add_argument("--max-edge-fill-gap", type=int, default=2)

    p.add_argument("--save-montages", type=int, default=12)
    p.add_argument("--save-tensors", type=int, default=0)
    return p.parse_args()


def read_segments(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["segment_id"]),
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                )
            )
    return rows


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance_norm(a: np.ndarray, b: np.ndarray, fw: int, fh: int) -> float:
    acx = (float(a[0]) + float(a[2])) * 0.5
    acy = (float(a[1]) + float(a[3])) * 0.5
    bcx = (float(b[0]) + float(b[2])) * 0.5
    bcy = (float(b[1]) + float(b[3])) * 0.5
    diag = float(np.hypot(fw, fh))
    return float(np.hypot(acx - bcx, acy - bcy) / max(diag, 1.0))


def predict_frame(model: YOLO, frame: np.ndarray, args) -> list[Detection]:
    kwargs = dict(
        source=frame,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=[0],
        max_det=40,
        verbose=False,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    result = model.predict(**kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    return [
        Detection(np.asarray(box, dtype=np.float32), float(conf))
        for box, conf in zip(boxes, confs)
    ]


def associate_window(
    detections_by_frame: list[list[Detection]],
    fw: int,
    fh: int,
    args,
) -> list[Tracklet]:
    """
    Very short-horizon MOT:
    - start tracklets from frame 0,
    - Hungarian assignment to active tracklets,
    - unmatched detections spawn new tracklets,
    - tracks can survive up to max_gap missing frames.
    """
    tracklets: dict[int, Tracklet] = {}
    next_tid = 1

    for pos, detections in enumerate(detections_by_frame):
        # Active tracks that were seen recently.
        active = []
        for tr in tracklets.values():
            if tr.last_pos >= 0 and (pos - tr.last_pos) <= args.max_gap + 1:
                active.append(tr)

        matched_det_idx = set()

        if active and detections:
            cost = np.full((len(active), len(detections)), 1e6, dtype=np.float32)

            for i, tr in enumerate(active):
                _, last_box = tr.last_box_before(pos)
                if last_box is None:
                    continue

                for j, det in enumerate(detections):
                    iou = box_iou(last_box, det.bbox)
                    dist = center_distance_norm(last_box, det.bbox, fw, fh)

                    # Allow either small overlap OR sufficiently small motion.
                    if iou < args.min_iou_gate and dist > args.max_center_dist:
                        continue

                    cost[i, j] = (
                        args.iou_weight * (1.0 - iou)
                        + args.center_weight * dist
                    )

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e5:
                    continue
                tr = active[i]
                det = detections[j]
                tr.boxes[pos] = det.bbox
                tr.confs[pos] = det.conf
                matched_det_idx.add(j)

        # Spawn tracklets for unmatched detections.
        for j, det in enumerate(detections):
            if j in matched_det_idx:
                continue
            tr = Tracklet(tid=next_tid)
            tr.boxes[pos] = det.bbox
            tr.confs[pos] = det.conf
            tracklets[next_tid] = tr
            next_tid += 1

    return list(tracklets.values())


def fill_box_sequence(track: Tracklet, max_edge_gap: int):
    seq = [None] * T
    observed_positions = sorted(track.boxes)

    if not observed_positions:
        return seq

    for pos, box in track.boxes.items():
        seq[pos] = box.copy()

    # Internal linear interpolation.
    for left, right in zip(observed_positions[:-1], observed_positions[1:]):
        if right <= left + 1:
            continue
        b0 = track.boxes[left]
        b1 = track.boxes[right]
        for pos in range(left + 1, right):
            alpha = (pos - left) / float(right - left)
            seq[pos] = (1 - alpha) * b0 + alpha * b1

    # Short edge fill only.
    first = observed_positions[0]
    for pos in range(max(0, first - max_edge_gap), first):
        seq[pos] = track.boxes[first].copy()

    last = observed_positions[-1]
    for pos in range(last + 1, min(T, last + max_edge_gap + 1)):
        seq[pos] = track.boxes[last].copy()

    return seq


def track_score(track: Tracklet) -> float:
    """
    Prioritize temporal presence, then anchor-frame availability, then confidence.
    Presence dominates because B8 needs stable player crops over all 9 frames.
    """
    anchor_bonus = 1.5 if 0 in track.boxes else 0.0
    early_bonus = 0.5 if track.first_pos <= 1 else 0.0
    return (
        10.0 * track.observed
        + 3.0 * track.avg_conf
        + anchor_bonus
        + early_bonus
    )


def select_12(tracklets: list[Tracklet], min_track_frames: int):
    eligible = [t for t in tracklets if t.observed >= min_track_frames]
    eligible.sort(key=track_score, reverse=True)
    return eligible[:MAX_PLAYERS]


def anchor_x(seq):
    if seq[0] is not None:
        return float((seq[0][0] + seq[0][2]) * 0.5)
    for box in seq:
        if box is not None:
            return float((box[0] + box[2]) * 0.5)
    return float("inf")


def clamp_box(box, fw, fh):
    if box is None:
        return None
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(int(round(x1)), fw - 1))
    y1 = max(0, min(int(round(y1)), fh - 1))
    x2 = max(0, min(int(round(x2)), fw))
    y2 = max(0, min(int(round(y2)), fh))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_tensor(frame_bgr, box):
    fh, fw = frame_bgr.shape[:2]
    b = clamp_box(box, fw, fh)
    if b is None:
        return None
    x1, y1, x2, y2 = b
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(rgb).crop((x1, y1, x2, y2))
    return CROP_TRANSFORM(crop)


def save_montage(frames, selected, seqs, out_path):
    tile_w, tile_h = 180, 150
    canvas = Image.new("RGB", (6 * tile_w, 2 * tile_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    anchor = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    fw, fh = anchor.size

    for slot in range(MAX_PLAYERS):
        x = (slot % 6) * tile_w
        y = (slot // 6) * tile_h

        if slot >= len(selected):
            draw.text((x + 5, y + 5), f"slot {slot} EMPTY", fill=(255,255,255))
            continue

        tr = selected[slot]
        box = seqs[tr.tid][0]
        observed = 0 in tr.boxes
        b = clamp_box(box, fw, fh)

        label = f"S{slot} T{tr.tid} {'OBS' if observed else 'FILL'} {tr.observed}/9"
        draw.text((x + 5, y + 5), label, fill=(255,255,255))

        if b is None:
            continue

        crop = anchor.crop(b)
        crop.thumbnail((tile_w - 8, tile_h - 34))
        bg = Image.new("RGB", (tile_w - 8, tile_h - 34), (0,0,0))
        bg.paste(crop, ((bg.width-crop.width)//2, (bg.height-crop.height)//2))
        canvas.paste(bg, (x + 4, y + 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
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
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    window_id = 0
    montage_saved = 0
    tensor_saved = 0

    print(f"Source: {source}")
    print(f"Resolution: {fw}x{fh}")
    print(f"imgsz={args.imgsz}, conf={args.conf}")
    print("Association: Hungarian IoU + center distance")
    print()

    for segment_id, seg_start, seg_end in segments:
        starts = list(range(seg_start, seg_end - T + 2, args.window_stride))

        for start in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames = []
            dets_by_frame = []
            valid = True

            for pos in range(T):
                ok, frame = cap.read()
                if not ok:
                    valid = False
                    break
                frames.append(frame.copy())
                dets_by_frame.append(predict_frame(model, frame, args))

            if not valid:
                continue

            tracklets = associate_window(dets_by_frame, fw, fh, args)
            selected = select_12(tracklets, args.min_track_frames)

            seqs = {tr.tid: fill_box_sequence(tr, args.max_edge_fill_gap) for tr in selected}
            selected.sort(key=lambda tr: anchor_x(seqs[tr.tid]))

            observed_cells = sum(tr.observed for tr in selected)
            effective_cells = sum(
                sum(box is not None for box in seqs[tr.tid]) for tr in selected
            )

            observed_cov = observed_cells / float(MAX_PLAYERS * T)
            effective_cov = effective_cells / float(MAX_PLAYERS * T)

            accepted = (
                len(selected) == MAX_PLAYERS
                and observed_cov >= args.min_observed_coverage
            )

            tensor = None
            if accepted:
                zero = torch.zeros(3, 224, 224)
                slot_tensors = []
                for tr in selected:
                    temporal = []
                    for pos in range(T):
                        crop = crop_tensor(frames[pos], seqs[tr.tid][pos])
                        temporal.append(crop if crop is not None else zero.clone())
                    slot_tensors.append(torch.stack(temporal, dim=0))
                tensor = torch.stack(slot_tensors, dim=0)
                if tuple(tensor.shape) != (12, 9, 3, 224, 224):
                    raise RuntimeError(tuple(tensor.shape))

            window_id += 1
            raw_counts = [len(x) for x in dets_by_frame]

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
                    "all_tracklets": len(tracklets),
                    "selected_tracks": len(selected),
                    "observed_cells": observed_cells,
                    "effective_cells": effective_cells,
                    "observed_coverage": round(observed_cov, 4),
                    "effective_coverage": round(effective_cov, 4),
                    "accepted": int(accepted),
                    "selected_track_observations": ",".join(str(t.observed) for t in selected),
                    "track_ids_left_to_right": ",".join(str(t.tid) for t in selected),
                }
            )

            if montage_saved < args.save_montages and selected:
                save_montage(
                    frames,
                    selected,
                    seqs,
                    montage_dir / f"window_{window_id:04d}_accepted_{int(accepted)}.jpg",
                )
                montage_saved += 1

            if accepted and tensor is not None and tensor_saved < args.save_tensors:
                tensor_dir.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, tensor_dir / f"window_{window_id:04d}.pt")
                tensor_saved += 1

        print(f"Segment {segment_id}: tested {len(starts)} windows")

    cap.release()

    csv_path = out_dir / "b8_hungarian_v3.csv"
    fields = [
        "window_id","segment_id","start_frame","key_frame","end_frame","key_time_sec",
        "mean_raw_detections","min_raw_detections","max_raw_detections",
        "all_tracklets","selected_tracks","observed_cells","effective_cells",
        "observed_coverage","effective_coverage","accepted",
        "selected_track_observations","track_ids_left_to_right"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if rows:
        raw = np.asarray([r["mean_raw_detections"] for r in rows], dtype=float)
        obs = np.asarray([r["observed_coverage"] for r in rows], dtype=float)
        eff = np.asarray([r["effective_coverage"] for r in rows], dtype=float)
        sel = np.asarray([r["selected_tracks"] for r in rows], dtype=int)
        acc = np.asarray([r["accepted"] for r in rows], dtype=int)

        print("\n===== B8 HUNGARIAN V3 SUMMARY =====")
        print(f"Windows tested: {len(rows)}")
        print(f"Mean raw YOLO detections/frame: {raw.mean():.3f}")
        print(f"Median raw YOLO detections/frame: {np.median(raw):.3f}")
        print(f"Mean observed coverage: {obs.mean():.3f}")
        print(f"Median observed coverage: {np.median(obs):.3f}")
        print(f"Mean effective coverage: {eff.mean():.3f}")
        print(f"Windows with 12 selected trajectories: {(sel == 12).mean():.1%}")
        print(f"Accepted windows: {acc.mean():.1%} ({acc.sum()}/{len(acc)})")
        if acc.sum():
            mask = acc == 1
            print(f"Accepted mean observed coverage: {obs[mask].mean():.3f}")
            print(f"Accepted median observed coverage: {np.median(obs[mask]):.3f}")
            print(f"Accepted mean effective coverage: {eff[mask].mean():.3f}")
        print(f"CSV: {csv_path.resolve()}")
        print(f"Montages: {montage_dir.resolve()}")

if __name__ == "__main__":
    main()
