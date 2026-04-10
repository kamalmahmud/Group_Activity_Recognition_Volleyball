from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

DATASET_ROOT = '/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos'

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


def load_video_annot(video_annot_path):
    with open(video_annot_path, 'r') as file:
        clip_category_dct = {}
        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]
    return clip_category_dct


class VolleyballDataset(Dataset):
    def __init__(self, root, video_ids, transform=None):
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
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
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
    )


# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.001

train_loader = build_dataloader('train', batch_size=batch_size, num_workers=4)
val_loader = build_dataloader('val', batch_size=batch_size, num_workers=4)
test_loader = build_dataloader('test', batch_size=batch_size, num_workers=4)

num_classes = 8
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
best_val_acc = 0.0
print('Training started')
for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

    # --- Validate ---
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f'Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f}')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_resnet50_volleyball.pth')
        print(f'  → Best model saved (val_acc={val_acc:.4f})')

print('Finished Training')
torch.save(model.state_dict(), 'trial1_resnet50_middle_frame.pth')
print('Final model saved as trial1_resnet50_middle_frame.pth')

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

class_names = list(GROUP_ACTIVITIES.keys())

print(f'F1 (macro):    {f1_score(all_labels, all_preds, average="macro"):.4f}')
print(f'F1 (weighted): {f1_score(all_labels, all_preds, average="weighted"):.4f}')
print()
print(classification_report(all_labels, all_preds, target_names=class_names))
