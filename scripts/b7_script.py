import torch
import torch.nn as nn
from data import GROUP_LABELS
from models.b7_model import B7Model
from scripts import device
from utils.runner import run

CLASS_NAMES = list(GROUP_LABELS.keys())

model = B7Model().to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Unwrap DataParallel to safely access submodules
raw_model = model.module if isinstance(model, nn.DataParallel) else model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {"params": raw_model.feature_extractor.parameters(), "lr": 1e-5},
    {"params": raw_model.player_lstm.parameters(), "lr": 3e-4},
    {"params": raw_model.frame_lstm.parameters(), "lr": 3e-4},
    {"params": raw_model.classifier.parameters(), "lr": 3e-4},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-6,
)

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
        num_workers=4,
        cm_filename="confusion_matrix_b7.png")
