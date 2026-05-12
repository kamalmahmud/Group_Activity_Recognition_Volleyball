import torch
import torch.nn as nn
from data import GROUP_LABELS, get_transform, get_data_loader
from models.b8_model import B8Model
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation
from utils.trainer import train

batch_size = 4
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

model = B8Model().to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

base_model = model.module if isinstance(model, nn.DataParallel) else model

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {"params": base_model.player_model.model.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
        {"params": base_model.player_model.lstm.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
        {"params": base_model.frame_lstm.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": base_model.frame_projection.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
        {"params": base_model.classifier.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
    ],
    betas=(0.9, 0.999),
    eps=1e-8,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-6,
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
        20,
        save_path,
    )

    full_evaluation(
        model,
        test_loader,
        criterion,
        device=device,
        class_names=CLASS_NAMES,
        cm_save_path=f"{save_path}confusion_matrix_b8.png",
    )