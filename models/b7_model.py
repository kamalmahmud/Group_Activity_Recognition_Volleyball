import torch
import torch.nn as nn

from models.b5_model import B5Model


class B7Model(nn.Module):
    def __init__(self, player_model: B5Model, hidden_size: int = 512, num_classes: int = 8,
                 freeze_backbone: bool = False, ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.player_model = player_model

        if freeze_backbone:
            for param in self.player_model.parameters():
                param.requires_grad = False

        player_hidden = player_model.lstm.hidden_size

        self.frame_lstm = nn.LSTM(
            input_size=player_hidden,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, mask=None):
        b, n, t, c, h, w = x.shape

        x = x.reshape(b * n, t, c, h, w)

        if self.freeze_backbone:
            self.player_model.eval()
            with torch.no_grad():
                player_out = self._player_lstm_all_steps(x)  # [B*N, T, H1]
        else:
            player_out = self._player_lstm_all_steps(x)

        player_out = player_out.reshape(b, n, t, -1)  # [B, N, T, H1]

        if mask is not None:
            mask = mask.to(device=player_out.device, dtype=torch.bool)
            neg_val = torch.finfo(player_out.dtype).min
            player_out = player_out.masked_fill(~mask.unsqueeze(-1), neg_val)

        frame_feats = player_out.max(dim=1).values  # [B, T, H1]

        lstm2_out, _ = self.frame_lstm(frame_feats)  # [B, T, hidden_size]
        out = lstm2_out[:, -1, :]  # [B, hidden_size]

        return self.classifier(out)  # [B, num_classes]

    def _player_lstm_all_steps(self, x_bn: torch.Tensor) -> torch.Tensor:
        bn, t, c, h, w = x_bn.shape
        player = x_bn.reshape(bn * t, c, h, w)

        frame_feats = self.player_model.model(player)
        frame_feats = frame_feats.reshape(bn, t, -1)  # [B*N, T, 2048]

        lstm_out, _ = self.lstm(frame_feats)  # [B*N, T, 512]
        return lstm_out
