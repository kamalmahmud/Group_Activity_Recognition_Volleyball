from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from ultralytics import YOLO

from deployment.b8_runtime_model import B8RuntimeModel


T = 9
MAX_PLAYERS = 12
KEY_POS = 5

CLASS_NAMES = [
    "l-pass",
    "r-pass",
    "l-spike",
    "r_spike",
    "l_set",
    "r_set",
    "l_winpoint",
    "r_winpoint",
]

DISPLAY_NAMES = {
    "l-pass": "Left Pass",
    "r-pass": "Right Pass",
    "l-spike": "Left Spike",
    "r_spike": "Right Spike",
    "l_set": "Left Set",
    "r_set": "Right Set",
    "l_winpoint": "Left Win Point",
    "r_winpoint": "Right Win Point",
}

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
    def observed(self):
        return len(self.boxes)

    @property
    def first_pos(self):
        return min(self.boxes) if self.boxes else 999

    @property
    def last_pos(self):
        return max(self.boxes) if self.boxes else -1

    @property
    def avg_conf(self):
        return float(np.mean(list(self.confs.values()))) if self.confs else 0.0

    def last_box_before(self, pos):
        pts = [p for p in self.boxes if p < pos]
        if not pts:
            return None
        return self.boxes[max(pts)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Raw volleyball video -> YOLO/Hungarian -> B8 predictions."
    )
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--b8-weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--output-dir", default="./runs/b8_predictions")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.03)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--device", default=None)
    p.add_argument("--window-stride", type=int, default=9)
    p.add_argument("--min-track-frames", type=int, default=6)
    p.add_argument("--min-observed-coverage", type=float, default=0.85)
    p.add_argument("--max-edge-fill-gap", type=int, default=2)
    p.add_argument("--max-center-dist", type=float, default=0.16)
    p.add_argument("--min-iou-gate", type=float, default=0.01)
    p.add_argument("--smooth-windows", type=int, default=3)
    p.add_argument("--save-annotated-video", action="store_true")
    return p.parse_args()


def read_segments(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                (
                    int(r["segment_id"]),
                    int(r["start_frame"]),
                    int(r["end_frame"]),
                )
            )
    return out


def load_b8(path, device):
    model = B8RuntimeModel()

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {
            "epoch": ckpt.get("epoch"),
            "val_acc": ckpt.get("val_acc"),
            "val_loss": ckpt.get("val_loss"),
        }
    else:
        state = ckpt
        meta = {}

    # Support DataParallel checkpoints if needed.
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, meta


def box_iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    ab = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def center_dist(a, b, fw, fh):
    ax = (float(a[0])+float(a[2]))/2
    ay = (float(a[1])+float(a[3]))/2
    bx = (float(b[0])+float(b[2]))/2
    by = (float(b[1])+float(b[3]))/2
    return float(np.hypot(ax-bx, ay-by) / max(np.hypot(fw, fh), 1.0))


def predict_yolo(model, frame, args):
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
        Detection(np.asarray(b, dtype=np.float32), float(c))
        for b, c in zip(boxes, confs)
    ]


def associate(dets_by_frame, fw, fh, args):
    tracks = {}
    next_tid = 1

    for pos, dets in enumerate(dets_by_frame):
        active = [
            t for t in tracks.values()
            if t.last_pos >= 0 and (pos - t.last_pos) <= 3
        ]

        matched = set()

        if active and dets:
            cost = np.full((len(active), len(dets)), 1e6, dtype=np.float32)

            for i, tr in enumerate(active):
                last = tr.last_box_before(pos)
                if last is None:
                    continue
                for j, det in enumerate(dets):
                    iou = box_iou(last, det.bbox)
                    dist = center_dist(last, det.bbox, fw, fh)
                    if iou < args.min_iou_gate and dist > args.max_center_dist:
                        continue
                    cost[i, j] = 0.65 * (1.0 - iou) + 0.35 * dist

            rr, cc = linear_sum_assignment(cost)
            for i, j in zip(rr, cc):
                if cost[i, j] >= 1e5:
                    continue
                tr = active[i]
                tr.boxes[pos] = dets[j].bbox
                tr.confs[pos] = dets[j].conf
                matched.add(j)

        for j, det in enumerate(dets):
            if j in matched:
                continue
            tr = Tracklet(next_tid)
            tr.boxes[pos] = det.bbox
            tr.confs[pos] = det.conf
            tracks[next_tid] = tr
            next_tid += 1

    return list(tracks.values())


def track_score(t):
    return (
        10.0 * t.observed
        + (1.5 if 0 in t.boxes else 0.0)
        + (0.5 if t.first_pos <= 1 else 0.0)
        + 3.0 * t.avg_conf
    )


def fill_sequence(track, edge_gap):
    seq = [None] * T
    pts = sorted(track.boxes)
    for p, b in track.boxes.items():
        seq[p] = b.copy()

    for left, right in zip(pts[:-1], pts[1:]):
        if right <= left + 1:
            continue
        a, b = track.boxes[left], track.boxes[right]
        for p in range(left+1, right):
            alpha = (p-left) / float(right-left)
            seq[p] = (1-alpha)*a + alpha*b

    if pts:
        first, last = pts[0], pts[-1]
        for p in range(max(0, first-edge_gap), first):
            seq[p] = track.boxes[first].copy()
        for p in range(last+1, min(T, last+edge_gap+1)):
            seq[p] = track.boxes[last].copy()

    return seq


def anchor_x(seq):
    if seq[0] is not None:
        b = seq[0]
        return float((b[0]+b[2])/2)
    for b in seq:
        if b is not None:
            return float((b[0]+b[2])/2)
    return float("inf")


def clamp_box(box, fw, fh):
    if box is None:
        return None
    x1,y1,x2,y2 = map(float, box)
    x1 = max(0, min(int(round(x1)), fw-1))
    y1 = max(0, min(int(round(y1)), fh-1))
    x2 = max(0, min(int(round(x2)), fw))
    y2 = max(0, min(int(round(y2)), fh))
    return None if x2 <= x1 or y2 <= y1 else (x1,y1,x2,y2)


def crop_tensor(frame, box):
    fh, fw = frame.shape[:2]
    b = clamp_box(box, fw, fh)
    if b is None:
        return None
    x1,y1,x2,y2 = b
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(rgb).crop((x1,y1,x2,y2))
    return CROP_TRANSFORM(crop)


def build_tensor(frames, tracklets, args):
    eligible = [t for t in tracklets if t.observed >= args.min_track_frames]
    eligible.sort(key=track_score, reverse=True)
    selected = eligible[:MAX_PLAYERS]

    seqs = {t.tid: fill_sequence(t, args.max_edge_fill_gap) for t in selected}
    selected.sort(key=lambda t: anchor_x(seqs[t.tid]))

    observed_cells = sum(t.observed for t in selected)
    effective_cells = sum(
        sum(b is not None for b in seqs[t.tid]) for t in selected
    )
    observed_cov = observed_cells / float(MAX_PLAYERS*T)
    effective_cov = effective_cells / float(MAX_PLAYERS*T)

    if len(selected) != 12 or observed_cov < args.min_observed_coverage:
        return None, observed_cov, effective_cov, selected

    zero = torch.zeros(3,224,224)
    slots = []

    for t in selected:
        temporal = []
        for pos in range(T):
            crop = crop_tensor(frames[pos], seqs[t.tid][pos])
            temporal.append(crop if crop is not None else zero.clone())
        slots.append(torch.stack(temporal, dim=0))

    tensor = torch.stack(slots, dim=0)
    return tensor, observed_cov, effective_cov, selected


def infer(model, tensor, device):
    x = tensor.unsqueeze(0).to(device)

    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return idx, probs


def smooth_predictions(rows, k):
    if k <= 1:
        for r in rows:
            r["smoothed_class"] = r["predicted_class"]
            r["smoothed_confidence"] = r["confidence"]
        return

    for i, row in enumerate(rows):
        same_segment = []
        for j in range(max(0, i-k+1), i+1):
            if rows[j]["segment_id"] == row["segment_id"]:
                same_segment.append(rows[j])

        avg = np.mean(
            np.asarray([x["_probs"] for x in same_segment], dtype=np.float32),
            axis=0,
        )
        idx = int(np.argmax(avg))
        row["smoothed_class"] = CLASS_NAMES[idx]
        row["smoothed_confidence"] = round(float(avg[idx]), 6)


def write_annotated_video(source, output, rows, fps):
    preds = sorted(rows, key=lambda r: r["key_frame"])
    if not preds:
        return

    cap = cv2.VideoCapture(str(source))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (fw,fh),
    )

    pred_idx = 0
    current = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        while pred_idx < len(preds) and preds[pred_idx]["key_frame"] <= frame_idx:
            current = preds[pred_idx]
            pred_idx += 1

        if current is not None:
            label = DISPLAY_NAMES[current["smoothed_class"]]
            conf = current["smoothed_confidence"]
            text = f"{label}  {conf:.1%}"
            cv2.rectangle(frame, (12, 12), (390, 65), (0,0,0), -1)
            cv2.putText(
                frame, text, (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,255,255), 2, cv2.LINE_AA
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device not in (None, "0") else
        ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device: {device}")

    yolo = YOLO(args.yolo_weights)
    b8, ckpt_meta = load_b8(args.b8_weights, device)

    print("B8 checkpoint loaded successfully.")
    if ckpt_meta:
        print(f"Checkpoint metadata: {ckpt_meta}")

    segments = read_segments(args.segments)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    predictions = []
    tested = 0
    skipped = 0

    for segment_id, seg_start, seg_end in segments:
        starts = list(range(seg_start, seg_end-T+2, args.window_stride))

        for start in starts:
            tested += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames = []
            dets_by_frame = []
            good = True

            for _ in range(T):
                ok, frame = cap.read()
                if not ok:
                    good = False
                    break
                frames.append(frame.copy())
                dets_by_frame.append(predict_yolo(yolo, frame, args))

            if not good:
                skipped += 1
                continue

            tracklets = associate(dets_by_frame, fw, fh, args)
            tensor, obs_cov, eff_cov, selected = build_tensor(frames, tracklets, args)

            if tensor is None:
                skipped += 1
                continue

            pred_idx, probs = infer(b8, tensor, device)
            key_frame = start + KEY_POS

            row = {
                "segment_id": segment_id,
                "start_frame": start,
                "key_frame": key_frame,
                "end_frame": start + 8,
                "key_time_sec": round(key_frame / fps, 3),
                "observed_coverage": round(float(obs_cov), 4),
                "effective_coverage": round(float(eff_cov), 4),
                "predicted_class": CLASS_NAMES[pred_idx],
                "display_name": DISPLAY_NAMES[CLASS_NAMES[pred_idx]],
                "confidence": round(float(probs[pred_idx]), 6),
                **{
                    f"p_{CLASS_NAMES[i]}": round(float(probs[i]), 6)
                    for i in range(len(CLASS_NAMES))
                },
                "_probs": probs.tolist(),
            }
            predictions.append(row)

        print(
            f"Segment {segment_id}: tested={len(starts)} "
            f"| accepted_predictions={sum(r['segment_id']==segment_id for r in predictions)}"
        )

    cap.release()

    smooth_predictions(predictions, args.smooth_windows)

    csv_path = out_dir / "predictions.csv"
    json_path = out_dir / "predictions.json"

    csv_fields = [
        "segment_id","start_frame","key_frame","end_frame","key_time_sec",
        "observed_coverage","effective_coverage",
        "predicted_class","display_name","confidence",
        "smoothed_class","smoothed_confidence",
        *[f"p_{name}" for name in CLASS_NAMES],
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in predictions:
            w.writerow({k: r[k] for k in csv_fields})

    json_rows = []
    for r in predictions:
        clean = {k:v for k,v in r.items() if k != "_probs"}
        clean["probabilities"] = {
            name: r[f"p_{name}"] for name in CLASS_NAMES
        }
        for name in CLASS_NAMES:
            clean.pop(f"p_{name}", None)
        json_rows.append(clean)

    payload = {
        "source": str(args.source),
        "fps": fps,
        "window_length_frames": 9,
        "key_position": 5,
        "window_stride": args.window_stride,
        "windows_tested": tested,
        "windows_predicted": len(predictions),
        "windows_skipped": skipped,
        "checkpoint": ckpt_meta,
        "predictions": json_rows,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n===== END-TO-END B8 INFERENCE =====")
    print(f"Windows tested: {tested}")
    print(f"Windows predicted: {len(predictions)}")
    print(f"Windows skipped: {skipped}")
    if tested:
        print(f"Prediction acceptance: {len(predictions)/tested:.1%}")

    if predictions:
        counts = {}
        for r in predictions:
            counts[r["smoothed_class"]] = counts.get(r["smoothed_class"], 0) + 1
        print("\nSmoothed class counts:")
        for name, count in sorted(counts.items(), key=lambda x:x[1], reverse=True):
            print(f"  {name:>12}: {count}")

        avg_conf = float(np.mean([r["confidence"] for r in predictions]))
        print(f"\nMean raw softmax confidence: {avg_conf:.3f}")

        print("\nFirst 12 predictions:")
        for r in predictions[:12]:
            print(
                f"{r['key_time_sec']:>7.2f}s | "
                f"{r['predicted_class']:<12} {r['confidence']:.3f} | "
                f"smooth={r['smoothed_class']:<12} {r['smoothed_confidence']:.3f} | "
                f"cov={r['observed_coverage']:.3f}"
            )

    if args.save_annotated_video and predictions:
        video_path = out_dir / "annotated_predictions.mp4"
        write_annotated_video(args.source, video_path, predictions, fps)
        print(f"Annotated video: {video_path.resolve()}")

    print(f"CSV:  {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
