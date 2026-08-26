from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackStats:
    first_frame: int
    last_frame: int
    detections: int = 0
    conf_sum: float = 0.0
    max_conf: float = 0.0
    min_conf: float = 1.0
    x_center_sum: float = 0.0
    y_center_sum: float = 0.0
    area_sum: float = 0.0

    def update(
        self,
        frame_idx: int,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_w: int,
        frame_h: int,
    ) -> None:
        self.last_frame = frame_idx
        self.detections += 1
        self.conf_sum += confidence
        self.max_conf = max(self.max_conf, confidence)
        self.min_conf = min(self.min_conf, confidence)

        xc = ((x1 + x2) / 2.0) / max(frame_w, 1)
        yc = ((y1 + y2) / 2.0) / max(frame_h, 1)
        area = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)

        self.x_center_sum += xc
        self.y_center_sum += yc
        self.area_sum += area

    @property
    def avg_conf(self) -> float:
        return self.conf_sum / max(self.detections, 1)

    @property
    def avg_x_center(self) -> float:
        return self.x_center_sum / max(self.detections, 1)

    @property
    def avg_y_center(self) -> float:
        return self.y_center_sum / max(self.detections, 1)

    @property
    def avg_box_area(self) -> float:
        return self.area_sum / max(self.detections, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track volleyball players with the fine-tuned YOLO detector + ByteTrack. "
            "Produces an annotated video, per-frame counts, and per-track statistics."
        )
    )
    parser.add_argument("--weights", required=True, help="Path to trained YOLO best.pt")
    parser.add_argument("--source", required=True, help="Input MP4/AVI path")
    parser.add_argument(
        "--output-dir",
        default="./runs/player_tracking",
        help="Directory for the annotated video and CSV summaries",
    )
    parser.add_argument(
        "--tracker",
        default=str(Path(__file__).with_name("bytetrack_volleyball.yaml")),
        help="Tracker YAML. Defaults to the volleyball ByteTrack config beside this script.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help=(
            "Detector confidence passed into ByteTrack. Keep low so ByteTrack can use "
            "lower-confidence boxes during temporary occlusion."
        ),
    )
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of recent center points drawn for each track; 0 disables trails.",
    )
    parser.add_argument(
        "--min-summary-frames",
        type=int,
        default=5,
        help="Only include tracks seen this many frames in the persistent-track summary.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0 or cpu. Default uses Ultralytics auto-selection.",
    )
    return parser.parse_args()


def make_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v is broadly available in Kaggle/Colab OpenCV builds.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer


def draw_track(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track_id: int,
    confidence: float,
    trail: deque[tuple[int, int]],
) -> None:
    x1, y1, x2, y2 = box

    # Draw a consistent, readable label. Avoid class label because there is only one class.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"ID {track_id}  {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    label_y1 = max(0, y1 - th - baseline - 6)
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + tw + 8, y1),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    if len(trail) >= 2:
        pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (255, 255, 255), 2)


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    output_dir = Path(args.output_dir)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = output_dir / f"{source.stem}_tracked.mp4"
    frame_csv_path = output_dir / "frame_tracking_counts.csv"
    track_csv_path = output_dir / "track_summary.csv"

    writer = make_writer(output_video, fps, width, height)

    trail_length = max(args.trail_length, 0)
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=max(trail_length, 1))
    )
    stats: dict[int, TrackStats] = {}
    frame_rows: list[dict] = []

    frame_idx = 0
    print(f"Source: {source}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Tracker: {args.tracker}")
    print(f"Detector conf: {args.conf}")
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        track_kwargs = dict(
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
            track_kwargs["device"] = args.device

        result = model.track(**track_kwargs)[0]

        tracked_this_frame = 0
        raw_detections = len(result.boxes) if result.boxes is not None else 0

        if (
            result.boxes is not None
            and result.boxes.is_track
            and result.boxes.id is not None
        ):
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            tracked_this_frame = len(track_ids)

            for xyxy, confidence, track_id in zip(
                boxes, confidences, track_ids
            ):
                x1, y1, x2, y2 = map(float, xyxy)
                confidence = float(confidence)
                track_id = int(track_id)

                xi1 = max(0, min(int(round(x1)), width - 1))
                yi1 = max(0, min(int(round(y1)), height - 1))
                xi2 = max(0, min(int(round(x2)), width - 1))
                yi2 = max(0, min(int(round(y2)), height - 1))

                if xi2 <= xi1 or yi2 <= yi1:
                    continue

                center = ((xi1 + xi2) // 2, (yi1 + yi2) // 2)
                trails[track_id].append(center)

                if track_id not in stats:
                    stats[track_id] = TrackStats(
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                    )

                stats[track_id].update(
                    frame_idx=frame_idx,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_w=width,
                    frame_h=height,
                )

                draw_track(
                    frame=frame,
                    box=(xi1, yi1, xi2, yi2),
                    track_id=track_id,
                    confidence=confidence,
                    trail=trails[track_id] if trail_length > 0 else deque(),
                )

        time_sec = frame_idx / fps
        frame_rows.append(
            {
                "frame": frame_idx,
                "time_sec": round(time_sec, 3),
                "raw_detections": raw_detections,
                "tracked_players": tracked_this_frame,
            }
        )

        cv2.putText(
            frame,
            f"Tracked players: {tracked_this_frame}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 250 == 0:
            print(
                f"Processed {frame_idx}/{total_frames if total_frames > 0 else '?'} "
                f"frames | unique track IDs so far: {len(stats)}"
            )

    cap.release()
    writer.release()

    with frame_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time_sec",
                "raw_detections",
                "tracked_players",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    track_rows = []
    for track_id, s in stats.items():
        if s.detections < args.min_summary_frames:
            continue

        span_frames = s.last_frame - s.first_frame + 1
        track_rows.append(
            {
                "track_id": track_id,
                "first_frame": s.first_frame,
                "last_frame": s.last_frame,
                "first_time_sec": round(s.first_frame / fps, 3),
                "last_time_sec": round(s.last_frame / fps, 3),
                "detections": s.detections,
                "span_frames": span_frames,
                "coverage_in_span": round(s.detections / max(span_frames, 1), 4),
                "avg_conf": round(s.avg_conf, 4),
                "min_conf": round(s.min_conf, 4),
                "max_conf": round(s.max_conf, 4),
                "avg_x_center_norm": round(s.avg_x_center, 4),
                "avg_y_center_norm": round(s.avg_y_center, 4),
                "avg_box_area_norm": round(s.avg_box_area, 6),
            }
        )

    track_rows.sort(key=lambda x: x["detections"], reverse=True)

    with track_csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "track_id",
            "first_frame",
            "last_frame",
            "first_time_sec",
            "last_time_sec",
            "detections",
            "span_frames",
            "coverage_in_span",
            "avg_conf",
            "min_conf",
            "max_conf",
            "avg_x_center_norm",
            "avg_y_center_norm",
            "avg_box_area_norm",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(track_rows)

    print("\nDone.")
    print(f"Annotated video: {output_video.resolve()}")
    print(f"Per-frame counts: {frame_csv_path.resolve()}")
    print(f"Track summary: {track_csv_path.resolve()}")
    print(f"Unique track IDs (all): {len(stats)}")
    print(
        f"Persistent tracks (>= {args.min_summary_frames} detections): "
        f"{len(track_rows)}"
    )

    print("\nTop persistent tracks:")
    for row in track_rows[:20]:
        print(
            f"ID {row['track_id']:>4} | "
            f"detections={row['detections']:>5} | "
            f"coverage={row['coverage_in_span']:.3f} | "
            f"avg_conf={row['avg_conf']:.3f} | "
            f"x={row['avg_x_center_norm']:.3f} | "
            f"y={row['avg_y_center_norm']:.3f}"
        )


if __name__ == "__main__":
    main()
