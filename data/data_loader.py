from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from volleyball_annot_loader import load_video_annot

DATASET_ROOT = '/kaggle/input/group-activity-recognition-volleyball/videos'

GROUP_ACTIVITIES = {
            'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
            'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
        }

IDX_TO_ACTIVITY = {v: k for k, v in GROUP_ACTIVITIES.items()}

TRAIN_VIDEOS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_VIDEOS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_VIDEOS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



class VolleyballDataset(Dataset):
    def __init__(self, root: str, video_ids, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for vid in video_ids:
            vid_dir = self.root / str(vid)
            ann_path = vid_dir / 'annotations.txt'

            if not vid_dir.exists() or not ann_path.exists():
                continue

            clip_category_dct = load_video_annot(str(ann_path))

            for clip_id, category in clip_category_dct.items():
                img_path = vid_dir / clip_id / f'{clip_id}.jpg'
                if not img_path.exists():
                    continue
                self.samples.append((str(img_path), GROUP_ACTIVITIES[category]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transform(split):
    if split == 'train':
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def build_dataloader(split, batch_size=32, num_workers=4):
    video_ids = {'train': TRAIN_VIDEOS, 'val': VAL_VIDEOS, 'test': TEST_VIDEOS}[split]

    dataset = VolleyballDataset(
        root=DATASET_ROOT,
        video_ids=video_ids,
        transform=build_transform(split),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
    )
