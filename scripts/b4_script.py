import torch
import torch.nn as nn
from torch.optim import AdamW
from data import GROUP_LABELS
from models.b4_model import B4Model
from scripts import  device
from utils.runner import run

checkpoint_path = "/kaggle/input/models/kamalalqedra/baseline4/pytorch/default/1/best_model.pth"
lr = 0.0001
CLASS_NAMES = GROUP_LABELS.keys()

model = B4Model(num_classes=len(CLASS_NAMES)).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
        mode="temporal_frame",
        num_epochs=20,
        batch_size=64,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b4.png")
