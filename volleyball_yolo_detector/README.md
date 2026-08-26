# Volleyball YOLO Player Detector

Add the `players_detection/` directory to the root of the existing
`Group_Activity_Recognition_Volleyball` repository.

Run commands from the repository root.

## 1. Install

```bash
pip install -r players_detection/requirements-yolo.txt
```

## 2. Build the YOLO dataset

Example for the paths already documented by the project on Kaggle:

```bash
python players_detection/prepare_yolo_dataset.py \
  --pkl /kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl \
  --videos /kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos \
  --output /kaggle/working/yolo_volleyball \
  --frame-stride 4 \
  --link-mode symlink
```

The output will be:

```text
yolo_volleyball/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 3. Train

```bash
python players_detection/train_yolo.py \
  --data /kaggle/working/yolo_volleyball/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project /kaggle/working/runs/player_detector \
  --name yolov8n_volleyball
```

Best weights:

```text
/kaggle/working/runs/player_detector/yolov8n_volleyball/weights/best.pt
```

## 4. Test on an arbitrary MP4

```bash
python players_detection/test_detector.py \
  --weights /kaggle/working/runs/player_detector/yolov8n_volleyball/weights/best.pt \
  --source /kaggle/working/test.mp4 \
  --conf 0.25
```

For a first run, keep the B8 video split intact. Do not train the detector
on the held-out B8 test videos if you want an honest end-to-end evaluation.
