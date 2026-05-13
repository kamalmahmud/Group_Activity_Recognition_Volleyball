import torch
import torch.nn as nn
from torch.optim import AdamW
from data import PLAYER_LABELS
from models.b3_a_model import B3AModel
from scripts import device
from utils.runner import run

lr = 1e-4
CLASS_NAMES = PLAYER_LABELS.keys()

model = B3AModel(num_classes=9).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(model.parameters(), lr=lr)
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
        mode="person",
        num_epochs=10,
        batch_size=32,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b3_a.png")
