import torch
import torch.nn as nn
from torch.optim import AdamW
from models import B1Model
from data import get_data_loader
from data import get_transform
from utils import train, full_evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pkl_path = ""
videos_path = ""
mode = "frame"
batch_size = 64
num_workers = 4
lr = 1e-3
CLASS_NAMES = [
    "l-pass", "r-pass", "l-spike", "r-spike",
    "l-set", "r-set", "l-winpoint", "r-winpoint"
]

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode=mode,
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

# ── Model / Loss / Optimizer ─────────────────────────────────────────────
model = B1Model(num_classes=8)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
model, history = train(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    CLASS_NAMES,
    30,
    "checkpoints", )

full_evaluation(model, test_loader,
                criterion,
                device=device,
                class_names=CLASS_NAMES,
                cm_save_path="volleyball_project/saves/b1")
