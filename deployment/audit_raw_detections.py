from __future__ import annotations
import argparse, csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25)

def read_segments(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["segment_id"]), int(row["start_frame"]), int(row["end_frame"])))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--output-dir", default="./runs/detection_audit")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=None)
    p.add_argument("--sample-stride", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    segments = read_segments(args.segments)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    print(f"Source: {args.source}")
    print(f"Resolution: {fw}x{fh}")
    print(f"YOLO imgsz: {args.imgsz}")
    print(f"Gameplay segments: {len(segments)}")

    for segment_id, start, end in segments:
        n = 0
        for frame_idx in range(start, end + 1, args.sample_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            kwargs = dict(source=frame, conf=0.01, iou=0.70, imgsz=args.imgsz, classes=[0], verbose=False)
            if args.device is not None:
                kwargs["device"] = args.device

            result = model.predict(**kwargs)[0]
            if result.boxes is None or len(result.boxes) == 0:
                confs = np.empty((0,), dtype=np.float32)
                boxes = np.empty((0,4), dtype=np.float32)
            else:
                confs = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()

            row = {"segment_id": segment_id, "frame": frame_idx, "time_sec": round(frame_idx/fps, 3)}
            for t in THRESHOLDS:
                row[f"count_ge_{t:.2f}"] = int((confs >= t).sum())

            keep = confs >= 0.05
            kb = boxes[keep]
            if len(kb):
                xcs = ((kb[:,0] + kb[:,2]) / 2.0) / fw
                areas = ((kb[:,2]-kb[:,0])*(kb[:,3]-kb[:,1]))/(fw*fh)
                row["spread_ge_0.05"] = round(float(xcs.max()-xcs.min()) if len(xcs) >= 2 else 0.0, 4)
                row["median_area_ge_0.05"] = round(float(np.median(areas)), 6)
            else:
                row["spread_ge_0.05"] = 0.0
                row["median_area_ge_0.05"] = 0.0

            rows.append(row)
            n += 1
        print(f"Segment {segment_id}: audited {n} frames")

    cap.release()

    csv_path = out_dir / "raw_detection_audit.csv"
    fields = ["segment_id","frame","time_sec",*[f"count_ge_{t:.2f}" for t in THRESHOLDS],"spread_ge_0.05","median_area_ge_0.05"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n===== RAW YOLO DETECTION AUDIT =====")
    print(f"Frames audited: {len(rows)}")
    for t in THRESHOLDS:
        vals = np.asarray([r[f"count_ge_{t:.2f}"] for r in rows], dtype=float)
        print(f"conf >= {t:0.2f} | mean={vals.mean():5.2f} | median={np.median(vals):4.1f} | >=10={(vals>=10).mean():6.1%} | >=12={(vals>=12).mean():6.1%}")
    print(f"\nCSV: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
