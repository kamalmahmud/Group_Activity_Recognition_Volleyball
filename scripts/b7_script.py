import torch
import torch.nn as nn
from torch.optim import AdamW
from models.b5_model import B5Model
from data import GROUP_LABELS
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b7_model import B7Model
from scripts import pkl_path, videos_path, device, save_path
from scripts.script_constants import player_temporal_checkpoint_path
from utils.evaluator import full_evaluation
from utils.trainer import train

batch_size = 8
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
player_model = B5Model().to(device)
checkpoint = torch.load(player_temporal_checkpoint_path, map_location="cpu")
player_model.load_state_dict(checkpoint["model_state_dict"])

b7_check_point = "/content/best_model.pth"
model = B7Model(player_model,freeze_backbone=True).to(device)
checkpoint_b7 = torch.load(b7_check_point, map_location="cpu")
model.load_state_dict(checkpoint_b7["model_state_dict"])

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    [
        {"params": model.player_model.model.parameters(), "lr": 1e-5},  # pretrained ResNet50
        {"params": model.player_model.lstm.parameters(), "lr": 1e-5},   # pretrained player LSTM
        {"params": model.frame_lstm.parameters(), "lr": 1e-3},          # new frame LSTM
        {"params": model.classifier.parameters(), "lr": 1e-3},          # new classifier
    ],
    weight_decay=1e-4,
)
# optimizer = AdamW(
#     [p for p in model.parameters() if p.requires_grad],
#     lr=1e-3,
#     weight_decay=1e-4,
# )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

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
        cm_save_path=f"{save_path}confusion_matrix_b7.png",
    )
