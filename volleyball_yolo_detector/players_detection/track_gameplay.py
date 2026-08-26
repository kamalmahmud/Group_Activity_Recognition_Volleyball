from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Segment:
    segment_id: int
    start_frame: int
    end_frame: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class TrackStats:
    segment_id: int
    track_id: int
    first_frame: int
    last_frame: int
    detections: int = 0
    conf_sum: float = 0.0
    x_sum: float = 0.0
    y_sum: float = 0.0
    area_sum: float = 0.0

    def update(
        self,
        frame_idx: int,
        conf: float,
        xyxy: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> None:
        x1, y1, x2, y2 = map(float, xyxy)
        self.last_frame = frame_idx
        self.detections += 1
        self.conf_sum += conf
        self.x_sum += ((x1 + x2) / 2.0) / frame_w
        self.y_sum += ((y1 + y2) / 2.0) / frame_h
        self.area_sum += ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)

    def row(self, fps: float) -> dict:
        span = self.last_frame - self.first_frame + 1
        return {
            "segment_id": self.segment_id,
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_time_sec": round(self.first_frame / fps, 3),
            "last_time_sec": round(self.last_frame / fps, 3),
            "detections": self.detections,
            "span_frames": span,
            "coverage_in_span": round(self.detections / max(span, 1), 4),
            "avg_conf": round(self.conf_sum / max(self.detections, 1), 4),
            "avg_x_center_norm": round(self.x_sum / max(self.detections, 1), 4),
            "avg_y_center_norm": round(self.y_sum / max(self.detections, 1), 4),
            "avg_box_area_norm": round(self.area_sum / max(self.detections, 1), 6),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect wide-court gameplay shots and track players within each segment."
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output-dir", default="./runs/gameplay_tracking")
    p.add_argument(
        "--tracker",
        default=str(Path(__file__).with_name("bytetrack_volleyball.yaml")),
    )
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)

    # Gameplay-view heuristics.
    p.add_argument("--min-players", type=int, default=7)
    p.add_argument("--max-players", type=int, default=18)
    p.add_argument("--min-horizontal-spread", type=float, default=0.42)
    p.add_argument("--max-median-box-area", type=float, default=0.075)

    # Temporal hysteresis. This prevents a brief occlusion from cutting a rally.
    p.add_argument("--start-confirm-frames", type=int, default=5)
    p.add_argument("--end-confirm-frames", type=int, default=10)
    p.add_argument("--min-segment-frames", type=int, default=20)
    p.add_argument("--trail-length", type=int, default=15)
    return p.parse_args()


def reset_ultralytics_tracker(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return
    for tracker in trackers:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def frame_geometry(
    boxes_xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[int, float, float]:
    """Return count, normalized horizontal spread, and median normalized area."""
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return 0, 0.0, 1.0

    x_centers = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0) / frame_w
    areas = (
        (boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
        * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        / (frame_w * frame_h)
    )

    spread = float(np.max(x_centers) - np.min(x_centers)) if len(x_centers) >= 2 else 0.0
    median_area = float(np.median(areas))
    return len(boxes_xyxy), spread, median_area


def is_gameplay_candidate(
    count: int,
    horizontal_spread: float,
    median_area: float,
    args: argparse.Namespace,
) -> bool:
    """
    Wide gameplay normally has many relatively small players distributed
    across a large fraction of the frame. Close-ups fail one or more tests.
    """
    count_ok = args.min_players <= count <= args.max_players
    spread_ok = horizontal_spread >= args.min_horizontal_spread
    area_ok = median_area <= args.max_median_box_area
    return count_ok and spread_ok and area_ok


def draw_box(
    frame: np.ndarray,
    xyxy: np.ndarray,
    track_id: int,
    conf: float,
    trail: deque,
) -> None:
    x1, y1, x2, y2 = map(int, np.round(xyxy))
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1] - 1)
    y2 = min(y2, frame.shape[0] - 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"P{track_id} {conf:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if len(trail) >= 2:
        pts = np.asarray(trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (255, 255, 255), 2)


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(weights)
    if not source.exists():
        raise FileNotFoundError(source)

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_out = out_dir / f"{source.stem}_gameplay_tracked.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_out}")

    frame_rows = []
    segments: list[Segment] = []
    track_stats: dict[tuple[int, int], TrackStats] = {}
    trails = defaultdict(lambda: deque(maxlen=args.trail_length))

    in_gameplay = False
    good_run = 0
    bad_run = 0
    current_segment_id = 0
    current_segment_start = None

    print(f"Source: {source}")
    print(f"Resolution: {frame_w}x{frame_h}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print()

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

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

        boxes_np = np.empty((0, 4), dtype=np.float32)
        conf_np = np.empty((0,), dtype=np.float32)
        ids = []

        if result.boxes is not None and len(result.boxes):
            boxes_np = result.boxes.xyxy.cpu().numpy()
            conf_np = result.boxes.conf.cpu().numpy()
            if result.boxes.is_track and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()

        count, spread, median_area = frame_geometry(
            boxes_np, frame_w, frame_h
        )
        candidate = is_gameplay_candidate(
            count, spread, median_area, args
        )

        if candidate:
            good_run += 1
            bad_run = 0
        else:
            bad_run += 1
            good_run = 0

        # Start only after several consecutive convincing wide-court frames.
        if not in_gameplay and good_run >= args.start_confirm_frames:
            in_gameplay = True
            current_segment_id += 1
            current_segment_start = frame_idx - good_run + 1
            trails.clear()

        # Record track stats only while the frame currently qualifies as gameplay.
        if in_gameplay and candidate and ids:
            for xyxy, conf, track_id in zip(boxes_np, conf_np, ids):
                key = (current_segment_id, int(track_id))
                if key not in track_stats:
                    track_stats[key] = TrackStats(
                        segment_id=current_segment_id,
                        track_id=int(track_id),
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                    )
                track_stats[key].update(
                    frame_idx,
                    float(conf),
                    xyxy,
                    frame_w,
                    frame_h,
                )

                x1, y1, x2, y2 = xyxy
                center = (
                    int(round((x1 + x2) / 2.0)),
                    int(round((y1 + y2) / 2.0)),
                )
                trails[(current_segment_id, int(track_id))].append(center)
                draw_box(
                    frame,
                    xyxy,
                    int(track_id),
                    float(conf),
                    trails[(current_segment_id, int(track_id))],
                )

        # A sustained non-gameplay run closes the segment and resets identity state.
        if in_gameplay and bad_run >= args.end_confirm_frames:
            segment_end = frame_idx - bad_run
            if (
                current_segment_start is not None
                and segment_end >= current_segment_start
                and (segment_end - current_segment_start + 1) >= args.min_segment_frames
            ):
                segments.append(
                    Segment(
                        segment_id=current_segment_id,
                        start_frame=current_segment_start,
                        end_frame=segment_end,
                    )
                )

            in_gameplay = False
            current_segment_start = None
            good_run = 0
            bad_run = 0
            trails.clear()
            reset_ultralytics_tracker(model)

        status = "GAMEPLAY" if (in_gameplay and candidate) else "SKIP"
        status_color = (0, 255, 0) if status == "GAMEPLAY" else (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"players={count} spread={spread:.2f} area={median_area:.3f}",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if in_gameplay:
            cv2.putText(
                frame,
                f"segment={current_segment_id}",
                (18, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

        frame_rows.append(
            {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "detections": count,
                "horizontal_spread": round(spread, 4),
                "median_box_area_norm": round(median_area, 6),
                "candidate": int(candidate),
                "in_gameplay": int(in_gameplay),
                "segment_id": current_segment_id if in_gameplay else "",
            }
        )

        frame_idx += 1
        if frame_idx % 250 == 0:
            print(
                f"Processed {frame_idx}/{total_frames if total_frames > 0 else '?'} "
                f"| segments confirmed so far: {len(segments)}"
            )

    # Close a gameplay segment that reaches EOF.
    if in_gameplay and current_segment_start is not None:
        end = frame_idx - 1
        if (end - current_segment_start + 1) >= args.min_segment_frames:
            segments.append(
                Segment(
                    segment_id=current_segment_id,
                    start_frame=current_segment_start,
                    end_frame=end,
                )
            )

    cap.release()
    writer.release()

    frames_csv = out_dir / "frame_gameplay_metrics.csv"
    with frames_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "frame",
            "time_sec",
            "detections",
            "horizontal_spread",
            "median_box_area_norm",
            "candidate",
            "in_gameplay",
            "segment_id",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)

    segments_csv = out_dir / "gameplay_segments.csv"
    with segments_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "segment_id",
            "start_frame",
            "end_frame",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "frames",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in segments:
            w.writerow(
                {
                    "segment_id": s.segment_id,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "start_time_sec": round(s.start_frame / fps, 3),
                    "end_time_sec": round(s.end_frame / fps, 3),
                    "duration_sec": round(s.n_frames / fps, 3),
                    "frames": s.n_frames,
                }
            )

    valid_segment_ids = {s.segment_id for s in segments}
    track_rows = [
        s.row(fps)
        for s in track_stats.values()
        if s.segment_id in valid_segment_ids
    ]
    track_rows.sort(
        key=lambda r: (r["segment_id"], -r["detections"])
    )

    tracks_csv = out_dir / "segment_track_summary.csv"
    with tracks_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "segment_id",
            "track_id",
            "first_frame",
            "last_frame",
            "first_time_sec",
            "last_time_sec",
            "detections",
            "span_frames",
            "coverage_in_span",
            "avg_conf",
            "avg_x_center_norm",
            "avg_y_center_norm",
            "avg_box_area_norm",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(track_rows)

    print("\nDone.")
    print(f"Annotated video: {video_out.resolve()}")
    print(f"Frame metrics:    {frames_csv.resolve()}")
    print(f"Gameplay segments:{segments_csv.resolve()}")
    print(f"Segment tracks:   {tracks_csv.resolve()}")
    print(f"Confirmed gameplay segments: {len(segments)}")

    for s in segments:
        n_tracks = len({
            r["track_id"] for r in track_rows
            if r["segment_id"] == s.segment_id
        })
        print(
            f"Segment {s.segment_id:>2}: "
            f"{s.start_frame:>5}-{s.end_frame:<5} "
            f"({s.n_frames / fps:>5.2f}s) | tracks={n_tracks}"
        )


if __name__ == "__main__":
    main()
