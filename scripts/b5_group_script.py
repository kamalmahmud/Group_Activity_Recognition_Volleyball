import torch
import torch.nn as nn
from torch.optim import AdamW
from data import GROUP_LABELS
from models.b5_group_classifier import B5BModel
from models.b5_model import B5Model
from scripts import device
from scripts.script_constants import player_temporal_checkpoint_path
from utils.runner import run

CLASS_NAMES = list(GROUP_LABELS.keys())

player_model = B5Model().to(device)
checkpoint = torch.load(player_temporal_checkpoint_path, map_location="cpu")
player_model.load_state_dict(checkpoint["model_state_dict"])

model = B5BModel(player_model=player_model, freeze_backbone=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW([
        {"params": model.player_model.model.parameters(), "lr": 1e-5},  # pretrained ResNet50
        {"params": model.player_model.lstm.parameters(), "lr": 1e-4},   # pretrained player LSTM
        {"params": model.group_classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

if __name__ == "__main__":
    run(
        model=model,
        mode="temporal_person_clip",
        num_epochs=20,
        batch_size=4,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b5_group.png")
