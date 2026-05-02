from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class B4Model(nn.Module):
    def __init__(self, num_classes=8, hidden_size=512, num_layers=1):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.temp = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)

        x = self.feature_extractor(x)  # [B*T, 2048, 1, 1]
        x = x.flatten(1)  # [B*T, 2048]

        # Restore temporal dimension
        x = x.reshape(b, t, 2048)  # [B, T, 2048]

        # Temporal modeling
        x, _ = self.temp(x)  # [B, T, hidden_size]

        # Use last frame representation
        x = x[:, -1, :]  # [B, hidden_size]

        x = self.fc(x)  # [B, num_classes]
        return x
