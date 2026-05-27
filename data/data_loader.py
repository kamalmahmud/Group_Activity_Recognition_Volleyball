from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from .dataset import VolleyballDataset


def make_weighted_sampler(dataset):
    labels = [sample["target"] for sample in dataset.samples]

    class_counts = Counter(labels)
    print("Train class counts:", class_counts)

    sample_weights = [1.0 / class_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def get_data_loader(pkl_path,
                    videos_path,
                    mode: str,
                    frame_transform,
                    crop_transform,
                    batch_size: int,
                    num_workers: int,
                    ):
    train_dataset = VolleyballDataset(pkl_path,
                                      videos_path,
                                      split="train",
                                      mode=mode,
                                      frame_transform=frame_transform,
                                      crop_transform=crop_transform
                                      )

    val_dataset = VolleyballDataset(pkl_path,
                                    videos_path,
                                    split="val",
                                    mode=mode,
                                    frame_transform=frame_transform,
                                    crop_transform=crop_transform)

    test_dataset = VolleyballDataset(pkl_path,
                                     videos_path,
                                     split="test",
                                     mode=mode,
                                     frame_transform=frame_transform,
                                     crop_transform=crop_transform)

    # train_sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              # sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True,)

    val_loader = DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True,)

    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True, prefetch_factor=2,
                             persistent_workers=True, )

    print(f"Train samples: {len(train_dataset.samples)}")
    print(f"Val samples:   {len(val_dataset.samples)}")
    print(f"Test samples:  {len(test_dataset.samples)}")

    return train_loader, val_loader, test_loader
