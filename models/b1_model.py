from torch import nn
from torchvision import models


class B1Model(nn.Module):
    def __init__(self, num_classes=8):
        super(B1Model, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
