import torch
import torch.nn as nn
from torch.optim import AdamW
from data import PLAYER_LABELS
from models.b5_model import B5Model
from scripts import device
from utils.runner import run

lr = 0.0001
CLASS_NAMES = PLAYER_LABELS.keys()

model = B5Model(num_classes=len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        mode="temporal_person",
        num_epochs=20,
        batch_size=64,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b5_a.png")
