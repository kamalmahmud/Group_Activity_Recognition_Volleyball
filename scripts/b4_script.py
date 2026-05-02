import torch
import torch.nn as nn
from torch.optim import AdamW

from data.data_loader import get_data_loader
from data.transformers import get_transform

from models.b4_model import B4Model
from utils.evaluator import full_evaluation
from utils.trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pkl_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl"
videos_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos"
save = "/kaggle/working/"
lr = 0.0001
batch_size = 32
num_workers = 4
CLASS_NAMES = [
    "l-pass", "r-pass", "l-spike", "r-spike",
    "l-set", "r-set", "l-winpoint", "r-winpoint"
]

frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="temporal_frame",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

model = B4Model(num_classes=len(CLASS_NAMES))
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(model.parameters(),lr=lr, weight_decay=1e-4)

if __name__ == "__main__":
    model, history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        CLASS_NAMES,
        25,
        save, )

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
                    cm_save_path='/kaggle/working/confusion_matrix.png')
