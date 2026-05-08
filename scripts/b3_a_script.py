import torch
import torch.nn as nn
from torch.optim import AdamW

from data import PLAYER_LABELS
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b3_a_model import B3AModel
from scripts import pkl_path, videos_path, device, save_path

from utils.evaluator import full_evaluation
from utils.trainer import train


checkpoint_path = "/kaggle/input/models/kamalalqedra/resnet50-player-classifier/pytorch/2/2/best_model.pth"
batch_size = 32
num_workers = 4
lr = 1e-4

CLASS_NAMES = PLAYER_LABELS.keys()

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="person",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

# ── Model / Loss / Optimizer ─────────────────────────────────────────────
model = B3AModel(num_classes=9).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print("Loaded checkpoint successfully")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Val acc:", checkpoint.get("val_acc"))
print("Val loss:", checkpoint.get("val_loss"))
print("Model device:", next(model.parameters()).device)
# Unfreeze only layer4 and fc
for name, param in model.model.named_parameters():
    if name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Check trainable layers
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(
    [
        {"params": model.model.layer4.parameters(), "lr": 5e-6},
        {"params": model.model.fc.parameters(), "lr": 2e-5},
    ],
    weight_decay=1e-4
)

if __name__ == "__main__":
    model, history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        CLASS_NAMES,
        5,
        save_path, )
    best_stage2_path = "/kaggle/working/best_model.pth"

    best_checkpoint = torch.load(best_stage2_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)

    print("Loaded best Stage 2 checkpoint")
    print("Best epoch:", best_checkpoint.get("epoch"))
    print("Best val acc:", best_checkpoint.get("val_acc"))
    print("Best val loss:", best_checkpoint.get("val_loss"))

    full_evaluation(
        model,
        test_loader,
        criterion,
        device=device,
        class_names=CLASS_NAMES,
        cm_save_path='{save_path}confusion_matrix.png'
    )
