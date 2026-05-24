import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class B6Model(nn.Module):
    def __init__(
            self,
            num_classes=8,
            cnn_feature_size=2048,
            lstm_hidden_size=2048,
            lstm_num_layers=2,
    ):
        super().__init__()

        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,  # 2048
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Linear(in_features=lstm_hidden_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):
        # x shape: [B, N, T, C, H, W]
        b, n, t, c, h, w = x.shape

        x = x.reshape(b * n * t, c, h, w)

        features = self.feature_extractor(x)  # [B*t*N, 2048]

        features = features.reshape(b, n, t, -1)  # [B, N, T, 2048]

        frame_features = torch.max(features, dim=1).values  # [B, t, 2048]

        lstm_out, _ = self.lstm(frame_features)  # [B, t, lstm_hidden_size]

        clip_features = lstm_out[:, -1, :]  # [B, lstm_hidden_size]

        logits = self.classifier(clip_features)  # [B, 8]

        return logits
