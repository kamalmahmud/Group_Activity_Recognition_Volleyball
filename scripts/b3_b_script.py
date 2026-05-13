import torch
import torch.nn as nn
from torch.optim import AdamW
from data import GROUP_LABELS
from models.b3_b_model import B3BModel
from scripts import device
from utils.runner import run

checkpoint_path = "/kaggle/input/models/kamalalqedra/resnet50-player-classifier/pytorch/2/3/best_model.pth"
lr = 1e-4
CLASS_NAMES = GROUP_LABELS.keys()

model = B3BModel(ckpt_path=checkpoint_path, num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.classifier.parameters(), lr=lr)
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
        mode="frame_person",
        num_epochs=20,
        batch_size=64,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=CLASS_NAMES,
        cm_filename="confusion_matrix_b3_b.png")
