from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class B3Model(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)