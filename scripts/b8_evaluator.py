import torch
import torch.nn as nn
from data import GROUP_LABELS, get_transform, get_data_loader
from models.b8_model import B8Model
from scripts import pkl_path, videos_path, device, save_path
from utils.evaluator import full_evaluation

checkpoint_path = "/kaggle/input/models/kamalalqedra/b8/pytorch/default/1/best_model.pth"
batch_size = 4
num_workers = 4
CLASS_NAMES = list(GROUP_LABELS.keys())
frame_transform, crop_transform = get_transform()
train_loader, val_loader, test_loader = get_data_loader(
    pkl_path=pkl_path,
    videos_path=videos_path,
    mode="temporal_person_clip",
    frame_transform=frame_transform,
    crop_transform=crop_transform,
    batch_size=batch_size,
    num_workers=num_workers,
)

model = B8Model().to(device)
criterion = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

base_model = model.module if isinstance(model, nn.DataParallel) else model

checkpoint = torch.load(checkpoint_path, map_location=device)

state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

base_model.load_state_dict(state_dict)

if __name__ == "__main__":

    full_evaluation(
        model,
        test_loader,
        criterion,
        device=device,
        class_names=CLASS_NAMES,
        cm_save_path=f"{save_path}confusion_matrix_b8.png",
    )