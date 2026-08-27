from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class B8RuntimeModel(nn.Module):
    """
    Deployment-equivalent B8 model.

    ResNet50 is intentionally created with weights=None: all learned weights are
    restored from the B8 checkpoint, so deployment never needs an ImageNet download.

    Hidden dimensions are configurable because older/local B8 checkpoints may
    have been trained before the current repository defaults were finalized.
    """

    def __init__(
        self,
        num_classes: int = 8,
        player_hidden_size: int = 2048,
        frame_hidden_size: int = 1024,
        classifier_hidden_size: int = 256,
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
            nn.Linear(frame_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden_size, num_classes),
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
