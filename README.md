# Group Activity Recognition — Volleyball

PyTorch implementation and experiments for **group activity recognition in volleyball videos**, inspired by the CVPR
2016 paper [A Hierarchical Deep Temporal Model for Group Activity Recognition](https://arxiv.org/abs/1511.06040).

The project explores a sequence of baselines that move from frame-level classification to player-level temporal modeling
and hierarchical group activity recognition.

---

## Highlights

- Uses the Volleyball Dataset with full-frame images, player bounding boxes, group activity labels, and individual
  player action labels.
- Implements multiple baselines: ResNet50 frame classification, player action classification, team-side pooling,
  temporal frame models, temporal player models, and hierarchical LSTM models.
- Provides reusable dataset modes through `VolleyballDataset`.
- Includes training, evaluation, classification reports, and row-normalized confusion matrix generation.
- Best reported model in this repository: **B8 Left/Right Hierarchical LSTM — 92% test accuracy**.

---

## Repository layout

```text
.
├── data/
│   ├── constants.py        # Train/val/test splits and label mappings
│   ├── dataset.py          # VolleyballDataset wrapper and mode dispatch
│   ├── data_loader.py      # DataLoader creation
│   ├── transformers.py     # Frame/crop transforms
│   ├── _builders.py        # Dataset index builders for each mode
│   ├── _getters.py         # __getitem__ logic for each mode
│   ├── _helpers.py         # Dataset utilities
│   ├── boxinfo.py          # BoxInfo annotation parser
│   └── pkl_data.txt        # Example annotation structure
├── models/
│   ├── b1_model.py
│   ├── b3_a_model.py
│   ├── b3_b_model.py
│   ├── b4_model.py
│   ├── b5_model.py
│   ├── b5_group_classifier.py
│   ├── b6_model.py
│   ├── b7_model.py
│   └── b8_model.py
├── scripts/
│   ├── script_constants.py # Dataset/checkpoint/save paths and device
│   ├── b1_script.py
│   ├── b3_a_script.py
│   ├── b3_b_script.py
│   ├── b4_script.py
│   ├── b5_a_script.py
│   ├── b5_group_script.py
│   ├── b6_script.py
│   ├── b7_script.py
│   ├── b8_script.py
│   └── b8_evaluator.py
├── utils/
│   ├── trainer.py          # AMP training loop and checkpointing
│   ├── evaluator.py        # Classification report and confusion matrix
│   └── runner.py           # Shared train/evaluate runner
├── results/                # Saved reports and confusion matrices
├── requirements.txt
└── LICENSE
```

---

## Dataset

This repository expects the Volleyball Dataset in the following layout:

```text
annot_all.pkl
videos/
└── <video_id>/
    └── <clip_id>/
        └── <frame_id>.jpg
```

The annotation pickle is expected to contain clip-level group labels and per-frame player boxes. The included
`data/pkl_data.txt` shows the structure:

```python
annotations[video_id][clip_id]["category"]
annotations[video_id][clip_id]["frame_boxes_dct"][frame_id] -> list[BoxInfo]
```

Each `BoxInfo` contains:

```text
player_ID, box=(x1, y1, x2, y2), frame_ID, lost, grouping, generated, category
```

The code uses the following fixed split IDs from `data/constants.py`:

| Split |                                                                                    Video IDs | Count |
|-------|---------------------------------------------------------------------------------------------:|------:|
| Train | `1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54` |    24 |
| Val   |                                    `0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51` |    15 |
| Test  |                                `4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47` |    16 |

### Labels

Group activity labels, exactly as defined in the code:

```python
GROUP_LABELS = {
    "l-pass": 0,
    "r-pass": 1,
    "l-spike": 2,
    "r_spike": 3,
    "l_set": 4,
    "r_set": 5,
    "l_winpoint": 6,
    "r_winpoint": 7,
}
```

Player action labels:

```python
PLAYER_LABELS = {
    "blocking": 0,
    "digging": 1,
    "falling": 2,
    "jumping": 3,
    "moving": 4,
    "setting": 5,
    "spiking": 6,
    "standing": 7,
    "waiting": 8,
}
```

---

## Dataset modes

`VolleyballDataset` supports six modes through the `mode` argument.

| Mode                   | Returned input                                | Target              | Used by         |
|------------------------|-----------------------------------------------|---------------------|-----------------|
| `frame`                | Single key frame `[C, H, W]`                  | Group label         | B1              |
| `person`               | Single player crop `[C, H, W]`                | Player action label | B3A             |
| `frame_person`         | 12 player crops `[12, C, H, W]`               | Group label         | B3B             |
| `temporal_frame`       | Full-frame sequence `[T, C, H, W]`            | Group label         | B4              |
| `temporal_person`      | One player track `[T, C, H, W]`               | Player action label | B5A             |
| `temporal_person_clip` | All player slots over time `[12, T, C, H, W]` | Group label         | B5B, B6, B7, B8 |

For `temporal_person_clip`, players are assigned to up to 12 fixed slots using the first frame sorted left-to-right.
Missing players are padded with zero crops.

---

## Results

The following results are taken from the classification reports committed under `results/`.

| Model | Task           | Main idea                                         | Test accuracy |
|-------|----------------|---------------------------------------------------|--------------:|
| B1    | Group activity | ResNet50 on one key frame                         |           78% |
| B3A   | Player action  | ResNet50 on individual player crops               |           74% |
| B3B   | Group activity | Frozen B3A features + left/right team pooling     |           89% |
| B4    | Group activity | ResNet50 frame features + temporal LSTM           |           75% |
| B5A   | Player action  | ResNet50 + player LSTM                            |           83% |
| B5B   | Group activity | B5 player features + max pooling over players     |           87% |
| B6    | Group activity | ResNet50 features + player pooling + 2-layer LSTM |           79% |
| B7    | Group activity | Two-level LSTM, pooling all players together      |           84% |
| B8    | Group activity | Two-level LSTM with left/right team pooling       |       **92%** |

Notes:

- `results/b5_report/Classification_Report.txt` contains both the B5 group report and a player-action report.
- `results/b3_reports/b3_b_report/Classification_Report.txt` contains multiple B3B experiments. The best listed variant
  is the left/right split with zero padding at 89% accuracy.
- B8 is the strongest reported variant because it keeps the left/right team structure while modeling temporal player
  features.

---

## Model summary

### B1 — frame classifier

File: `models/b1_model.py`  
Script: `scripts/b1_script.py`

A ResNet50 ImageNet backbone with a dropout + linear classification head for 8 group activity classes.

```text
frame -> ResNet50 -> Dropout -> Linear(2048 -> 8)
```

### B3A — player action classifier

File: `models/b3_a_model.py`  
Script: `scripts/b3_a_script.py`

A ResNet50 crop classifier for 9 individual player actions.

```text
player crop -> ResNet50 -> Linear(2048 -> 512) -> ReLU -> Dropout -> Linear(512 -> 9)
```

### B3B — player features to group activity

File: `models/b3_b_model.py`  
Script: `scripts/b3_b_script.py`

Loads a trained B3A checkpoint, removes its classification head, freezes the feature extractor, splits player crops into
left and right team slots, max-pools each side, concatenates both team features, and predicts the group activity.

```text
12 crops -> frozen B3A features [12, 2048]
        -> left max-pool [2048] + right max-pool [2048]
        -> concat [4096]
        -> MLP -> 8 group classes
```

### B4 — temporal frame classifier

File: `models/b4_model.py`  
Script: `scripts/b4_script.py`

Extracts ResNet50 features from every frame in a clip and applies an LSTM over the frame sequence.

```text
frames [T, C, H, W] -> ResNet50 per frame -> LSTM -> classifier
```

### B5A — temporal player action classifier

File: `models/b5_model.py`  
Script: `scripts/b5_a_script.py`

Applies ResNet50 to each crop in a player track, passes the sequence through an LSTM, concatenates CNN features with
LSTM outputs, and classifies the player action.

```text
player track [T, C, H, W]
    -> ResNet50 per crop [T, 2048]
    -> player LSTM [T, hidden]
    -> concat CNN + LSTM features
    -> player action classifier
```

### B5B — B5 features for group classification

File: `models/b5_group_classifier.py`  
Script: `scripts/b5_group_script.py`

Uses a `B5Model` as a player feature extractor, pools over the 12 player slots, and classifies the group activity.

```text
clip [12, T, C, H, W]
    -> B5 player features for each player
    -> max-pool over players
    -> group classifier
```

### B6 — temporal group model with player pooling

File: `models/b6_model.py`  
Script: `scripts/b6_script.py`

Extracts ResNet50 features for every player crop at every time step, max-pools over the player dimension for each frame,
then applies a 2-layer LSTM for group classification.

```text
clip [12, T, C, H, W]
    -> ResNet50 crop features [12, T, 2048]
    -> max-pool over players [T, 2048]
    -> LSTM(hidden=2048, layers=2)
    -> classifier
```

### B7 — hierarchical LSTM, all players pooled together

File: `models/b7_model.py`  
Script: `scripts/b7_script.py`

Builds a two-level temporal model. First, each player track is modeled by a player LSTM. Then all player features are
max-pooled per frame and passed into a frame-level LSTM.

```text
clip [12, T, C, H, W]
    -> ResNet50 per crop
    -> player LSTM per player
    -> max-pool over all 12 players per frame
    -> frame LSTM
    -> classifier
```

### B8 — hierarchical LSTM with left/right team pooling

File: `models/b8_model.py`  
Script: `scripts/b8_script.py`

The best reported model. It follows the hierarchical structure of B7 but preserves team-side structure by pooling slots
`0:6` and `6:12` separately before the frame-level LSTM.

```text
clip [12, T, C, H, W]
    -> ResNet50 per crop
    -> player LSTM per player
    -> left max-pool + right max-pool per frame
    -> concat team features
    -> frame LSTM
    -> classifier
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kamalmahmud/Group_Activity_Recognition_Volleyball.git
cd Group_Activity_Recognition_Volleyball
```

### 2. Create an environment

The repository records Python 3.10.19 in `requirements.txt`. Install Python through your environment manager first, then
install the Python packages.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies listed by the repository:

```text
torch ~= 2.10.0
torchvision ~= 0.25.0
pillow ~= 12.1.1
matplotlib ~= 3.10.8
scikit-learn ~= 1.7.2
tqdm ~= 4.67.3
```

### 3. Configure dataset and checkpoint paths

Paths are defined in `scripts/script_constants.py`.

On Kaggle, the code detects `KAGGLE_KERNEL_RUN_TYPE` and expects:

```text
/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl
/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos
/kaggle/working/
```

Outside Kaggle, it expects KaggleHub-style cached paths by default:

```text
/root/.cache/kagglehub/datasets/sherif31/group-activity-recognition-volleyball/versions/1/annot_all.pkl
/root/.cache/kagglehub/datasets/sherif31/group-activity-recognition-volleyball/versions/1/videos
/content/
```

For local training, edit `scripts/script_constants.py`, for example:

```python
pkl_path = "/path/to/annot_all.pkl"
videos_path = "/path/to/videos"
save_path = "./outputs/"
player_temporal_checkpoint_path = "./checkpoints/best_model_player.pth"
```

Checkpoint-dependent scripts also contain their own hard-coded checkpoint paths. Update them before running:

- `scripts/b3_b_script.py` expects a trained B3A checkpoint.
- `scripts/b4_script.py` loads a B4 checkpoint before training/evaluation.
- `scripts/b8_evaluator.py` expects a trained B8 checkpoint.

Model checkpoint files are not committed to the repository. The `.gitignore` excludes `*.pth` files.

---

## Running experiments

Run commands from the repository root. The existing scripts are written to be executed as files, for example
`python scripts/b1_script.py`.

| Experiment                      | Command                             | Dataset mode           |
|---------------------------------|-------------------------------------|------------------------|
| B1 frame classifier             | `python scripts/b1_script.py`       | `frame`                |
| B3A player classifier           | `python scripts/b3_a_script.py`     | `person`               |
| B3B group classifier            | `python scripts/b3_b_script.py`     | `frame_person`         |
| B4 temporal frame model         | `python scripts/b4_script.py`       | `temporal_frame`       |
| B5A temporal player model       | `python scripts/b5_a_script.py`     | `temporal_person`      |
| B5B group classifier            | `python scripts/b5_group_script.py` | `temporal_person_clip` |
| B6 temporal group model         | `python scripts/b6_script.py`       | `temporal_person_clip` |
| B7 hierarchical LSTM            | `python scripts/b7_script.py`       | `temporal_person_clip` |
| B8 left/right hierarchical LSTM | `python scripts/b8_script.py`       | `temporal_person_clip` |
| B8 checkpoint evaluation        | `python scripts/b8_evaluator.py`    | `temporal_person_clip` |

Each training script uses `utils.runner.run()`, which:

1. Builds transforms with `get_transform()`.
2. Creates train/val/test dataloaders.
3. Trains with `utils.trainer.train()`.
4. Saves the best validation checkpoint as `best_model.pth` under `save_path`.
5. Runs final test evaluation.
6. Saves a row-normalized confusion matrix image.

---

## Programmatic usage

```python
from data.dataset import VolleyballDataset
from data.transformers import get_transform

pkl_path = "/path/to/annot_all.pkl"
videos_path = "/path/to/videos"

frame_transform, crop_transform = get_transform()

dataset = VolleyballDataset(
    pkl_path=pkl_path,
    videos_path=videos_path,
    split="train",
    mode="temporal_person_clip",
    frame_transform=frame_transform,
    crop_transform=crop_transform,
)

x, y = dataset[0]
print(x.shape)  # [12, T, 3, 224, 224]
print(y)  # group activity label
```

---

## Training details

The shared training loop in `utils/trainer.py` uses:

- AdamW optimizers configured in each script.
- `ReduceLROnPlateau` schedulers in the scripts.
- CUDA automatic mixed precision with `torch.autocast` and `torch.amp.GradScaler`.
- Best-checkpoint saving based on validation accuracy.
- Test-set evaluation after reloading the best checkpoint.

The evaluation utilities in `utils/evaluator.py` provide:

- Accuracy via `sklearn.metrics.accuracy_score`.
- Full `classification_report`.
- Row-normalized confusion matrix saved as an image.

Transforms from `data/transformers.py`:

| Input type  | Transform                                                       |
|-------------|-----------------------------------------------------------------|
| Full frame  | Resize `256x256`, center crop `224x224`, ImageNet normalization |
| Player crop | Resize `224x224`, ImageNet normalization                        |

---

## Troubleshooting

### `FileNotFoundError` for `annot_all.pkl` or frames

Update `pkl_path` and `videos_path` in `scripts/script_constants.py`.

### Missing `.pth` checkpoint

The repository does not include model checkpoints. Train the required model first or update the hard-coded checkpoint
path in the relevant script.

### CUDA out of memory

Lower `batch_size` and/or `num_workers` in the script you are running. The temporal player-clip models are memory-heavy
because they process tensors shaped like `[B, 12, T, 3, 224, 224]`.

### Import errors

Run commands from the repository root, for example:

```bash
python scripts/b8_script.py
```

---

## Results files

Saved reports and confusion matrices are organized under:

```text
results/
├── b1_report/
├── b3_reports/
├── b4_report/
├── b5_report/
├── b6_report/
├── b7_report/
└── b8_report/
```

---

## Citation

If you use this project, cite the original paper:

```bibtex
@inproceedings{ibrahim2016hierarchical,
  title={A Hierarchical Deep Temporal Model for Group Activity Recognition},
  author={Ibrahim, Moustafa and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
```

Original paper/resources:

- [A Hierarchical Deep Temporal Model for Group Activity Recognition](https://arxiv.org/abs/1511.06040)
- [Original deep-activity-rec repository](https://github.com/mostafa-saad/deep-activity-rec)

---

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
