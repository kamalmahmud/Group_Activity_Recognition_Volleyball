import os.path
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from volleyball_annot_loader import load_video_annot

DATASET_ROOT = '/kaggle/input/group-activity-recognition-volleyball/videos'


class VolleyballDataset(Dataset):
    def __init__(self, videos_path: str, transforms=None, split='train'):
        self.videos_path = videos_path
        self.transform = transforms
        self.split = split
        self.splits = {
            'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }
        self.labels = {
            'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
            'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
        }
        self.frames_labels = []

        for folder in self.splits[self.split]:
            annot_path = os.path.join(self.videos_path, f'{folder}', 'annotations.txt')
            frame_dic = load_video_annot(annot_path)

            for frame, category in frame_dic.items():
                frame_path = os.path.join(self.videos_path, f'{folder}', frame, f'{frame}.jpg')
                if os.path.exists(frame_path):
                    self.frames_labels.append((frame_path, self.labels[category]))

    def __len__(self):
        return len(self.frames_labels)

    def __getitem__(self, idx):
        frame_path, label = self.frames_labels[idx]
        image = read_image(frame_path, mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        return image, label
