import torch
import torch.nn as nn
from data import GROUP_LABELS
from models.b5_group_classifier import B5BModel
from models.b5_model import B5Model
from scripts import device
from utils.runner import run

CLASS_NAMES = list(GROUP_LABELS.keys())

player_model = B5Model().to(device)
model = B5BModel(player_model=player_model, freeze_backbone=False).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

if __name__ == "__main__":
    run(
        model=model,
        mode="temporal_person_clip",
        num_epochs=15,
        batch_size=8,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        num_workers=12,
        cm_filename="confusion_matrix_b5_group.png")
