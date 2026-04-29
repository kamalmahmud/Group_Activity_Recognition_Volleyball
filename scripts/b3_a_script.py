import torch
import torch.nn as nn
from torch.optim import AdamW
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b3_model import B3AModel
from utils.evaluator import full_evaluation
from utils.trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/kaggle/input/models/kamalalqedra/resnet50-player-classifier/pytorch/2/1/best_model5.pth"
pkl_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl"
videos_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos"
save = "/kaggle/working/"
batch_size = 64
num_workers = 4
lr = 1e-4

CLASS_NAMES = [
    "blocking",
    "digging",
    "falling",
    "jumping",
    "moving",
    "setting",
    "spiking",
    "standing",
    "waiting",
]

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="person",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

# ── Model / Loss / Optimizer ─────────────────────────────────────────────
model = B3AModel(num_classes=9).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print("Loaded checkpoint successfully")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Val acc:", checkpoint.get("val_acc"))
print("Val loss:", checkpoint.get("val_loss"))
print("Model device:", next(model.parameters()).device)
# Freeze all ResNet layers except final classifier
for name, param in model.model.named_parameters():
    if name.startswith("fc"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Check trainable layers
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=1e-4
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
        3,
        save, )

    full_evaluation(
        model,
        test_loader,
        criterion,
        device=device,
        class_names=CLASS_NAMES,
        cm_save_path='/kaggle/working/confusion_matrix.png'
    )
