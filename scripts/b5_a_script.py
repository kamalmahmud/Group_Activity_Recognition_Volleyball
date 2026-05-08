import torch.nn as nn
from torch.optim import AdamW
from data import PLAYER_LABELS
from data.data_loader import get_data_loader
from data.transformers import get_transform
from models.b5_model import B5Model
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation
from utils.trainer import train

lr = 0.0001
batch_size = 16
num_workers = 8

CLASS_NAMES = PLAYER_LABELS.keys()
frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="temporal_person",
    frame_transform=frame_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    crop_transform=crop_transform,
)

model = B5Model(num_classes=len(CLASS_NAMES))
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        25,
        save_path, )

    full_evaluation(model, test_loader,
                    criterion,
                    device=device,
                    class_names=CLASS_NAMES,
                    cm_save_path='{save_path}confusion_matrix_b5_player.png')
