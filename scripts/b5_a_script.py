import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from data import PLAYER_LABELS
from models.b5_model import B5Model
from utils.runner import run


CLASS_NAMES = list(PLAYER_LABELS.keys())


def setup_ddp():
    """
    torchrun automatically provides:
    LOCAL_RANK, RANK, WORLD_SIZE
    """
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    local_rank, rank, world_size = setup_ddp()

    is_main_process = rank == 0

    model = B5Model(num_classes=len(CLASS_NAMES)).to(local_rank)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        # Use this only if some branches/parameters are sometimes unused.
        # find_unused_parameters=True,
    )

    optimizer = AdamW([
        {"params": model.module.model.parameters(), "lr": 1e-5},  # ResNet / backbone
        {"params": model.module.lstm.parameters(), "lr": 1e-4},   # LSTM
        {"params": model.module.person_classifier.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=[1e-7, 1e-6, 1e-6],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if is_main_process:
        print(f"Using DDP with {world_size} GPUs")

    try:
        run(
            model=model,
            mode="temporal_person",
            num_epochs=15,
            batch_size=32,  # this is now PER GPU unless your run() changes it
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            class_names=CLASS_NAMES,
            num_workers=4,
            cm_filename="confusion_matrix_b5_a.png",

            # Add these args to your run() function:
            device=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            world_size=world_size,
            is_main_process=is_main_process,
            distributed=True,
        )
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()