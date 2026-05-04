import torch
import torch.nn as nn
from torch.optim import AdamW

from data import GROUP_LABELS
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b1_model import B1Model
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation
from utils.trainer import train

batch_size = 64
num_workers = 4
lr = 1e-4
CLASS_NAMES = GROUP_LABELS.keys()

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="frame",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

# ── Model / Loss / Optimizer ─────────────────────────────────────────────
model = B1Model(num_classes=8)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)

if __name__ == "__main__":
    model, history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        CLASS_NAMES,
        30,
        save_path, )

    full_evaluation(model, test_loader,
                    criterion,
                    device=device,
                    class_names=CLASS_NAMES,
                    cm_save_path="/kaggle/working/confusion_matrix.png")
