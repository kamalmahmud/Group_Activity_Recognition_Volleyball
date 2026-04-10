from torch import nn
from torchvision import models
class B1_Model(nn.Module):
    def __init__(self, num_classes=8):
        super(B1_Model, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return self.model(x)