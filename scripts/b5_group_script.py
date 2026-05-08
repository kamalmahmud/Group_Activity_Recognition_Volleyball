import torch.nn as nn
import torch
from torch.optim import AdamW
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b5_model import B5Model
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation
from utils.trainer import train
from models.b5_group_classifier import B5BModel
from data import GROUP_LABELS

checkpoint_path = "/kaggle/input/models/kamalalqedra/temporal-player-action/pytorch/default/1/best_model.pth"
lr = 1e-3
batch_size = 16
num_workers = 4
CLASS_NAMES = list(GROUP_LABELS.keys())
frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="temporal_person_clip",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

player_model = B5Model().to(device)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
player_model.load_state_dict(checkpoint["model_state_dict"])

model = B5BModel(player_model=player_model, freeze_backbone=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW([
    {"params": model.player_model.parameters(), "lr": 1e-5},
    {"params": model.group_classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
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
        save_path, )

    full_evaluation(model, test_loader,
                    criterion,
                    device=device,
                    class_names=CLASS_NAMES,
                    cm_save_path='/kaggle/working/confusion_matrix.png')
