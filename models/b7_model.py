import torch
import torch.nn as nn

from models import B5Model


class B7Model(nn.Module):
    def __init__(self,player_model:B5Model,hidden_size: int = 2048, num_classes: int = 8,freeze_backbone: bool = True):
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.player_model = player_model

        if freeze_backbone:
            for param in self.player_model.parameters():
                param.requires_grad = False
            self.player_model.eval()

        player_hidden = player_model.fusion_dim

        self.frame_lstm = nn.LSTM(
            input_size=player_hidden,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape

        x = x.reshape(b * n, t, c, h, w)

        if self.freeze_backbone:
            with torch.no_grad():
                player_out = self.player_model(x, return_all_steps=True)
        else:
            player_out = self.player_model(x,return_all_steps=True)

        player_out = player_out.reshape(b, n, t, -1)  # [B, N, T, fusion_dim]

        frame_feats = player_out.max(dim=1).values  # [B, T, fusion_dim]

        lstm_out, _ = self.frame_lstm(frame_feats)  # [B, T, hidden_size]
        video_features = lstm_out[:, -1, :]  # [B, hidden_size]

        return self.classifier(video_features)  # [B, num_classes]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.player_model.eval()   # always keep frozen backbone in eval
        return self
