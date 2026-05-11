import torch
import torch.nn as nn
from torch.optim import AdamW

from data import GROUP_LABELS
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b6_model import B6Model
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation
from utils.trainer import train

checkpoint_path = "/kaggle/input/models/kamalalqedra/temporal-player-action/pytorch/default/1/best_model.pth"
lr = 1e-3
batch_size = 32
num_workers = 4
CLASS_NAMES = list(GROUP_LABELS.keys())

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="temporal_person_clip",
    frame_transform=frame_transform,
    crop_transform=crop_transform,
    batch_size=batch_size,
    num_workers=num_workers,
)

model = B6Model(ckpt_path=checkpoint_path, num_classes=8, freeze_backbone=True).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=lr,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
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
        scheduler,
        25,
        save_path,
    )

    full_evaluation(
        model,
        test_loader,
        criterion,
        device=device,
        class_names=CLASS_NAMES,
        cm_save_path=f"{save_path}confusion_matrix_b6.png",
    )