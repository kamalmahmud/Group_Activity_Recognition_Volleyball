from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the trained volleyball player detector on an image or video."
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Image/video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="./runs/player_detector_inference")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    if results:
        first = results[0]
        print(f"First frame/image detections: {len(first.boxes)}")

    output_dir = Path(args.project) / args.name
    print(f"Annotated output saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
