import torch
from torch import nn
from torchvision import models


class B5Model(nn.Module):
    def __init__(self, num_classes=9, hidden_size=1024, num_layers=1):
        # resnet50 and lstm for player classes then take last time step for all 12 players
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fusion_dim = in_features + hidden_size * 2  # 2048 + 2048
        self.person_classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim ),
            nn.Linear(self.fusion_dim , 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_features=False,return_all_steps=False):
        # [batch size, num frames, channels, height, width]
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        
        frame_features = self.model(x)
        # [batch size * num frames, 2048]

        frame_features = frame_features.reshape(b, t, -1)
        # [batch size, num frames, 2048]

        lstm_out, _ = self.lstm(frame_features)
        # [batch size, num frames, hidden_size * 2]

        combined_out = torch.cat((frame_features, lstm_out), dim=2)
        # [B, T, 2048 + hidden_size]
        if return_all_steps:
            return combined_out

        player_features,_ = combined_out.max(dim=1)
        # [batch size, 2048 + hidden_size]

        player_logits = self.person_classifier(player_features)
        # [B, num_person_classes]

        if return_features:
            return player_logits, player_features

        return player_logits
