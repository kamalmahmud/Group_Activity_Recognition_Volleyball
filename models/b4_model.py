from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class B4Model(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.temp = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=3, bidirectional=True)

        self.model.fc = nn.Linear(in_features=2048, out_features=8)
    def forward(self, x):
        x = self.feature_extractor(x)
        x, _ = self.temp(x)
        x = x.view(x.size(0), -1)
        