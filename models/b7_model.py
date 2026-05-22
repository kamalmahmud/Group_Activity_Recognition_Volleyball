import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class B7Model(nn.Module):
    def __init__(
        self,
        num_classes=8,
        player_hidden_size=2048,
        frame_hidden_size=1024,
    ):
        super(B7Model, self).__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.player_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=player_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.player_feat_dim = 2048 + player_hidden_size

        self.frame_lstm = nn.LSTM(
            input_size=self.player_feat_dim,
            hidden_size=frame_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(frame_hidden_size),
            nn.Linear(frame_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 12, T, 3, 224, 224]
        b, n, t, c, h, w = x.shape

        if n != 12:
            raise ValueError(f"B7Model expects 12 player slots, got {n}")

        x = x.reshape(b * n * t, c, h, w)

        cnn_feats = self.feature_extractor(x).flatten(1)
        # [B*12*T, 2048]

        cnn_seq = cnn_feats.reshape(b * n, t, 2048)
        # [B*12, T, 2048]

        player_lstm_out, _ = self.player_lstm(cnn_seq)
        # [B*12, T, player_hidden_size]

        player_seq = torch.cat([cnn_seq, player_lstm_out], dim=2)
        # [B*12, T, 2048 + player_hidden_size]

        player_seq = player_seq.reshape(b, n, t, self.player_feat_dim)
        # [B, 12, T, player_feat_dim]

        frame_feats = player_seq.max(dim=1).values
        # [B, T, player_feat_dim]

        frame_lstm_out, _ = self.frame_lstm(frame_feats)
        # [B, T, frame_hidden_size]

        logits = self.classifier(frame_lstm_out[:, -1])
        # [B, num_classes]

        return logits