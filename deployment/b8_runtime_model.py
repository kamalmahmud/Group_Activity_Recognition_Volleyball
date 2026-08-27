from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class B8RuntimeModel(nn.Module):
    """
    Same module structure as models/b8_model.py, but ResNet50 is created with
    weights=None so deployment never needs to download ImageNet weights.
    The trained checkpoint supplies every parameter through load_state_dict().
    """

    def __init__(
        self,
        num_classes: int = 8,
        player_hidden_size: int = 2048,
        frame_hidden_size: int = 1024,
    ):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.player_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=player_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.player_feat_dim = 2048 + player_hidden_size

        self.frame_lstm = nn.LSTM(
            input_size=self.player_feat_dim * 2,
            hidden_size=frame_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(frame_hidden_size),
            nn.Linear(frame_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 12, T, 3, 224, 224]
        b, n, t, c, h, w = x.shape
        if n != 12:
            raise ValueError(f"B8 expects 12 player slots, got {n}")

        x = x.reshape(b * n * t, c, h, w)
        cnn_feats = self.feature_extractor(x).flatten(1)

        cnn_seq = cnn_feats.reshape(b * n, t, 2048)
        player_lstm_out, _ = self.player_lstm(cnn_seq)

        player_seq = torch.cat([cnn_seq, player_lstm_out], dim=2)
        player_seq = player_seq.reshape(b, n, t, self.player_feat_dim)

        left_feats = player_seq[:, :6]
        right_feats = player_seq[:, 6:]

        left_pooled = left_feats.max(dim=1).values
        right_pooled = right_feats.max(dim=1).values

        frame_feats = torch.cat([left_pooled, right_pooled], dim=2)
        group_lstm_out, _ = self.frame_lstm(frame_feats)

        return self.classifier(group_lstm_out[:, -1])
