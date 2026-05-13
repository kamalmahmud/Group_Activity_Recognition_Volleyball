import torch
import torch.nn as nn
from torch.optim import AdamW

from data import GROUP_LABELS
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b3_b_model import B3BModel
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation
from utils.trainer import train


checkpoint_path = "/kaggle/input/models/kamalalqedra/resnet50-player-classifier/pytorch/2/3/best_model.pth"
batch_size = 64
num_workers = 4
lr = 1e-4
CLASS_NAMES = GROUP_LABELS.keys()

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="frame_person",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

model = B3BModel(ckpt_path=checkpoint_path, num_classes=8)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.classifier.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

if __name__ == "__main__":
    model, history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        CLASS_NAMES,
        scheduler,
        40,
        save_path, )

    best_stage2_path = "/kaggle/working/best_model.pth"

    best_checkpoint = torch.load(best_stage2_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)

    print("Best epoch:", best_checkpoint.get("epoch"))
    print("Best val acc:", best_checkpoint.get("val_acc"))
    print("Best val loss:", best_checkpoint.get("val_loss"))

    full_evaluation(model, test_loader,
                    criterion,
                    device=device,
                    class_names=CLASS_NAMES,
                    cm_save_path=save_path)
