from torch.utils.data import DataLoader
from .dataset import VolleyballDataset


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
                                    split="test",
                                    mode=mode,
                                    frame_transform=frame_transform,
                                    crop_transform=crop_transform)
    test_dataset = VolleyballDataset(pkl_path,
                                     videos_path,
                                     split="test",
                                     mode=mode,
                                     frame_transform=frame_transform,
                                     crop_transform=crop_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
