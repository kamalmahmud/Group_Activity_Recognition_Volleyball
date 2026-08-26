from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n for Volleyball Dataset player detection."
    )
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--project", default="./runs/player_detector")
    parser.add_argument("--name", default="yolov8n_volleyball")
    parser.add_argument(
        "--device",
        default=None,
        help="Examples: 0, cpu. Default automatically selects CUDA when available.",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Starting from: {args.model}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        pretrained=True,
        verbose=True,
        plots=True,
        close_mosaic=10,
    )

    best_path = (
        Path(args.project) / args.name / "weights" / "best.pt"
    )
    print(f"\nExpected best checkpoint: {best_path.resolve()}")

    if not best_path.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt was not found at {best_path}"
        )

    print("\nRunning validation with best.pt...")
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        device=device,
        plots=True,
    )

    print("\nValidation metrics")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\nRunning held-out test split...")
    test_metrics = best_model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        device=device,
        plots=False,
    )

    print("\nTest metrics")
    print(f"mAP50:     {test_metrics.box.map50:.4f}")
    print(f"mAP50-95:  {test_metrics.box.map:.4f}")
    print(f"Precision: {test_metrics.box.mp:.4f}")
    print(f"Recall:    {test_metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
