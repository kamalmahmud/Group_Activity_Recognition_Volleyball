import torch
import torch.nn as nn
from torch.optim import AdamW
from data import GROUP_LABELS
from models.b6_model import B6Model
from scripts import device
from utils.runner import run

checkpoint_path = "/kaggle/input/models/kamalmahmuod/b3-player/pytorch/default/1/best_model_b3_a.pth"
lr = 1e-4
CLASS_NAMES = list(GROUP_LABELS.keys())

model = B6Model(ckpt_path=checkpoint_path, num_classes=8).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=lr,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

if __name__ == "__main__":
    run(
        model=model,
        mode="temporal_person_clip",
        num_epochs=20,
        batch_size=16,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b6.png")
