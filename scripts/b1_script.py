import torch
import torch.nn as nn
from torch.optim import AdamW
from data import GROUP_LABELS
from models.b1_model import B1Model
from scripts import device
from utils.runner import run

lr = 1e-4
CLASS_NAMES = GROUP_LABELS.keys()

model = B1Model(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

if __name__ == "__main__":
    run(
        model=model,
        mode="frame",
        num_epochs=10,
        batch_size=128,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b1.png")
