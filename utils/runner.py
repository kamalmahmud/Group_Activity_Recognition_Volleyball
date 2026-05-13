from typing import List, Optional

import torch.nn as nn

from data.data_loader import get_data_loader
from data.transformers import get_transform
from data.constants import GROUP_LABELS
from scripts.script_constants import pkl_path, videos_path, save_path, device
from utils.trainer import train
from utils.evaluator import full_evaluation


def run(
    model: nn.Module,
    mode: str,
    num_epochs: int,
    optimizer,
    *,
    criterion: Optional[nn.Module] = None,
    scheduler=None,
    batch_size: int = 32,
    num_workers: int = 4,
    class_names: Optional[List[str]] = None,
    cm_filename: str = "confusion_matrix.png",
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if class_names is None:
        class_names = list(GROUP_LABELS.keys())

    frame_transform, crop_transform = get_transform()
    train_loader, val_loader, test_loader = get_data_loader(
        pkl_path=pkl_path,
        videos_path=videos_path,
        mode=mode,
        frame_transform=frame_transform,
        crop_transform=crop_transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model, history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        class_names,
        scheduler,
        num_epochs,
        save_path,
    )

    full_evaluation(
        model,
        test_loader,
        criterion,
        device=device,
        class_names=class_names,
        cm_save_path=f"{save_path}{cm_filename}",
    )


