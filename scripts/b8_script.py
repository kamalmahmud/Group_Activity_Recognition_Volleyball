import torch
import torch.nn as nn
from data import GROUP_LABELS
from models.b8_model import B8Model
from scripts import device
from utils.runner import run

CLASS_NAMES = list(GROUP_LABELS.keys())

model = B8Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

base_model = model.module if isinstance(model, nn.DataParallel) else model


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
        cm_filename="confusion_matrix_b8.png")
