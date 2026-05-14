# Group Activity Recognition — Volleyball

> **PyTorch re-implementation of**  
> *"A Hierarchical Deep Temporal Model for Group Activity Recognition"*  
> Ibrahim et al., CVPR
> 2016 · [Paper](https://arxiv.org/abs/1607.02643) · [Original repo](https://github.com/mostafa-saad/deep-activity-rec)

---

## Overview

This project implements a series of progressively more sophisticated baselines for **group activity recognition** in
volleyball videos. Starting from a simple single-frame ResNet classifier and building up to a full two-level
hierarchical LSTM model, each baseline is a self-contained experiment that mirrors the architectural ideas introduced in
the paper.

The core insight of the paper is that group activity can be understood **hierarchically**: first model what each
individual player is doing over time, then aggregate those per-player representations to classify the team-level
activity.

---

## Results Summary

| Baseline | Description                                     | Test Accuracy |
|----------|-------------------------------------------------|:-------------:|
| B1       | ResNet50 frame classifier                       |      78%      |
| B3A      | ResNet50 player action classifier               |      74%      |
| B3B      | Frozen B3A + left/right pooling → group         |      89%      |
| B4       | ResNet50 + LSTM temporal frame classifier       |      75%      |
| B5       | ResNet50 + LSTM temporal player classifier      |      73%      |
| B6       | Frozen B5 + player pool + frame LSTM → group    |      79%      |
| B7       | Two-level LSTM (player LSTM → frame LSTM)       |      84%      |
| **B8**   | **B5 all-steps + left/right pool + frame LSTM** |    **92%**    |

> **B8 achieves the best result at 92%**, demonstrating the value of preserving team-side structure and rich per-step
> player features throughout the temporal model.

---

## Dataset

**Volleyball Dataset** — introduced in the original paper.

| Property                    | Value                     |
|-----------------------------|---------------------------|
| Total videos                | 55 volleyball matches     |
| Total clips                 | 4,830                     |
| Frames per clip             | 41                        |
| Group activity classes      | 8                         |
| Individual action classes   | 9                         |
| Players annotated per frame | up to 12 (bounding boxes) |

**Splits**

| Split | Video IDs                                                                                  | Count |
|-------|--------------------------------------------------------------------------------------------|-------|
| Train | 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54 | 24    |
| Val   | 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51                                    | 15    |
| Test  | 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47                                | 16    |

**Group activity labels**

| Label                       | Description                         |
|-----------------------------|-------------------------------------|
| `l-pass` / `r-pass`         | Left / right team passing           |
| `l-spike` / `r-spike`       | Left / right team spiking           |
| `l_set` / `r_set`           | Left / right team setting           |
| `l_winpoint` / `r_winpoint` | Left / right team winning the point |

**Individual player action labels**

`blocking · digging · falling · jumping · moving · setting · spiking · standing · waiting`

---

## Project Structure

```text
.
├── data/
│   ├── constants.py        # Dataset splits and label mappings
│   ├── dataset.py          # VolleyballDataset class
│   ├── data_loader.py      # Train/val/test DataLoader creation
│   ├── transformers.py     # Image transforms for frames and crops
│   ├── _builders.py        # Index builders for each dataset mode
│   ├── _getters.py         # Item getters for each dataset mode
│   ├── _helpers.py         # Dataset helper methods
│   └── boxinfo.py          # BoxInfo parser
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
│   ├── b1_script.py
│   ├── b3_a_script.py
│   ├── b3_b_script.py
│   ├── b4_script.py
│   ├── b5_a_script.py
│   ├── b5_group_script.py
│   ├── b6_script.py
│   ├── b7_script.py
│   ├── b8_script.py
│   ├── b8_evaluator.py
│   └── script_constants.py
├── utils/
│   ├── trainer.py          # Training loop and checkpointing
│   ├── evaluator.py        # Classification report and confusion matrix
│   └── runner.py           # Shared training/evaluation runner
├── results/                # Saved reports and confusion matrices
├── requirements.txt
└── LICENSE
```

---

## Dataset Modes

`VolleyballDataset` supports six modes, selected via the `mode` argument:

| Mode                   | Input                                      | Target        | Used by         |
|------------------------|--------------------------------------------|---------------|-----------------|
| `frame`                | Single key frame                           | Group label   | B1              |
| `person`               | Single player crop (±1 frame offset)       | Player action | B3A             |
| `frame_person`         | Frame + all player crops                   | Group label   | B3B             |
| `temporal_frame`       | Sequence of full frames                    | Group label   | B4              |
| `temporal_person`      | Sequence of crops per player               | Player action | B5              |
| `temporal_person_clip` | All players × all frames `[N, T, C, H, W]` | Group label   | B5B, B6, B7, B8 |

---

## Baselines

The baselines form a deliberate progression — each one adds one architectural idea on top of the last.

---

### B1 — Frame-level ResNet50 · `78%`

**Architecture:** ResNet50 (ImageNet pre-trained) with a replaced FC head → 8 group classes.  
**Input:** Single key frame `[B, C, H, W]`.  
**Idea:** Simplest possible baseline. No temporal or person-level information.

```
ResNet50 → Dropout(0.5) → Linear(2048 → 8)
```

---

### B3A — Player Action Classifier · `74%`

**Architecture:** ResNet50 with a replaced FC head → 9 player action classes.  
**Input:** Individual player crop `[B, C, H, W]`.  
**Idea:** Learn to classify what each player is doing. Used as a pre-trained backbone for B3B. Note: accuracy is on the
*player-level* task (9 classes), not the group task.

```
ResNet50 → Dropout(0.2) → Linear(2048 → 9)
```

---

### B3B — Person-level Features → Group Classifier · `89%`

**Architecture:** Frozen B3A backbone; crops from left and right sides of the court are max-pooled separately,
concatenated, then classified.  
**Input:** All player crops for a frame `[B, 12, C, H, W]` (6 left, 6 right).  
**Idea:** Aggregate individual player features by team side. The large jump from B1 (78% → 89%) shows that person-level
features are far more discriminative than the raw frame alone.

```
Frozen ResNet50 → features[B, 12, 2048]
  → left max-pool[B, 2048]  +  right max-pool[B, 2048]
  → cat → [B, 4096]
  → Linear(4096→1024) → ReLU → Dropout → Linear(1024→8)
```

---

### B4 — Temporal Frame Model · `75%`

**Architecture:** ResNet50 extracts per-frame features; an LSTM models the temporal sequence; the mean hidden state is
classified.  
**Input:** Frame sequence `[B, T, C, H, W]`.  
**Idea:** Add temporal context without using bounding boxes. The modest result (75%) compared to B1 (78%) suggests that
temporal context alone — without person-level grounding — adds limited value.

```
ResNet50 (per frame) → [B, T, 2048]
  → LSTM(2048 → 512) → mean over T
  → Linear(512→512) → ReLU → Dropout → Linear(512→8)
```

---

### B5 — Temporal Player Action Classifier · `73%`

**Architecture:** ResNet50 + LSTM over a player's crop sequence; outputs a combined per-step feature (frame feature ∥
LSTM hidden state) as well as a classification logit.  
**Input:** Temporal crops for a single player `[B, T, C, H, W]`.  
**Idea:** Implements the *person-level LSTM* from the paper. The combined output `[2048 + 512]` is the rich per-player
representation passed to group models. Accuracy is on the *player-level* task (9 classes).

```
ResNet50 (per frame) → [B, T, 2048]
  → LSTM(2048 → 512) → [B, T, 512]
  → cat(frame_feats, lstm_out) → [B, T, 2560]
  → last step → Linear(2560 → 9)
```

---

### B6 — Player LSTM + Frame LSTM (Frozen B5) · `79%`

**Architecture:** Frozen B5 backbone extracts per-player, per-frame features; max-pooling over players produces a
per-frame scene representation; an LSTM models the temporal sequence.  
**Input:** `[B, 12, T, C, H, W]` with validity mask.  
**Idea:** Hierarchical temporal model with a frozen backbone. The frozen backbone limits the model's ability to adapt
player features for the group task.

```
Frozen B5 → features[B, N, T, 2048]
  → masked max-pool over N → frame_feats[B, T, 2048]
  → LSTM(2048 → 512) → last step
  → Linear(512→256) → ReLU → Dropout → Linear(256→8)
```

---

### B7 — Two-Level LSTM (End-to-End) · `84%`

**Architecture:** Like B6 but trained end-to-end with differential learning rates. The player-level LSTM outputs all
time steps, which are max-pooled over players per frame and fed to a second LSTM.  
**Input:** `[B, 12, T, C, H, W]` with validity mask.  
**Idea:** The full hierarchical temporal model from the paper, trained jointly. The 5-point gain over B6 (84% vs 79%)
confirms that fine-tuning the backbone end-to-end matters significantly.

```
ResNet50 (per player, per frame) → [B*N, T, 2048]
  → Player LSTM(2048 → 512) → [B*N, T, 512]   ← all steps
  → reshape → [B, N, T, 512]
  → masked max-pool over N → [B, T, 512]
  → Frame LSTM(512 → 512) → last step
  → Linear(512→256) → ReLU → Dropout → Linear(256→8)
```

Optimizer uses layered learning rates:

| Sub-module        | LR     |
|-------------------|--------|
| ResNet50 backbone | `1e-5` |
| Player LSTM       | `1e-4` |
| Frame LSTM        | `1e-3` |
| Classifier head   | `1e-3` |

---

### B8 — Left/Right Hierarchical LSTM · `92%`  Best

**Architecture:** B5 backbone returns combined features (frame + LSTM hidden) at every time step. Players are split into
left-team / right-team; each side is max-pooled over players per frame; the two sides are concatenated, projected, then
modeled with a frame-level LSTM.  
**Input:** `[B, 12, T, C, H, W]` with validity mask.  
**Idea:** Preserves team-side symmetry explicitly throughout the temporal model — the left/right split of B3B extended
across time with rich per-step player features. This is the best-performing model at 92%.

```
B5 (all steps) → [B*N, T, 2560]
  → reshape → [B, N, T, 2560]
  → left_pool[B, T, 2560]  +  right_pool[B, T, 2560]
  → cat → [B, T, 5120]
  → LayerNorm → Linear(5120→512) → ReLU → Dropout
  → Frame LSTM(512 → 512) → last step
  → Linear(512→256) → ReLU → Dropout → Linear(256→8)
```

---

## Training Details

All baselines share the same training infrastructure (`utils/trainer.py`, `utils/evaluator.py`, `utils/runner.py`).

| Setting         | Value                                                           |
|-----------------|-----------------------------------------------------------------|
| Optimizer       | AdamW                                                           |
| LR scheduler    | `ReduceLROnPlateau` (factor 0.5, patience 3)                    |
| Loss            | CrossEntropyLoss (label smoothing 0.1 on B3A, B4)               |
| Mixed precision | `torch.autocast` + `GradScaler` (CUDA)                          |
| Checkpointing   | Best val-accuracy model saved as `best_model.pth`               |
| Evaluation      | Row-normalised confusion matrix + sklearn classification report |

**Image pre-processing**

| Transform | Frame              | Player crop       |
|-----------|--------------------|-------------------|
| Resize    | 256×256            | 224×224           |
| Crop      | CenterCrop 224×224 | —                 |
| Normalize | ImageNet mean/std  | ImageNet mean/std |

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Data

Download the Volleyball Dataset from Kaggle:

```
sherif31/group-activity-recognition-volleyball
```

The dataset provides:

- `annot_all.pkl` — annotations pickle file
- `videos/` — frame images organised as `videos/<video_id>/<clip_id>/<frame_id>.jpg`

Paths are configured automatically for Kaggle vs. local environments in `scripts/script_constants.py`.

### Running a baseline

Each baseline has its own script. Example for B8:

```bash
python -m scripts.b8_script
```

To run the standalone B8 evaluator on a saved checkpoint:

```bash
python -m scripts.b8_evaluator
```

---

## Citation

```bibtex
@InProceedings{Ibrahim_2016_CVPR,
  author    = {Ibrahim, Mostafa S. and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2016}
}
```

---

## Acknowledgements

- Original paper and dataset: [mostafa-saad/deep-activity-rec](https://github.com/mostafa-saad/deep-activity-rec)
- Volleyball Dataset hosted on [Kaggle](https://www.kaggle.com/datasets/sherif31/group-activity-recognition-volleyball)