import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class B8Model(nn.Module):
    def __init__(self,num_classes=8,hidden_size=1024):
        super(B8Model, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.player_lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.player_feat_dim = 2048 + hidden_size

        self.frame_lstm = nn.LSTM(
            input_size=self.player_feat_dim * 2,
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

    def _masked_group_max(self, feats, mask):
        """
        feats: [B, S, T, D]
        mask:  [B, S, T]
        returns: [B, T, D]
        """
        mask = mask.unsqueeze(-1)  # [B, S, T, 1]
        neg_value = torch.finfo(feats.dtype).min

        feats = feats.masked_fill(~mask, neg_value)
        pooled = feats.max(dim=1).values  # [B, T, D]

        # If a whole side is missing at a timestep, replace -inf-like values with zeros
        valid_any = mask.any(dim=1)  # [B, T, 1]
        pooled = torch.where(valid_any, pooled, torch.zeros_like(pooled))

        return pooled

    def forward(self,x,mask=None):
        # batch, players, time, channels, height, width
        b,n,t,c,h,w = x.shape

        if mask is None:
            # Detect zero-padded crops if mask was not provided
            mask = x.flatten(3).abs().sum(dim=3) > 0  # [B, N, T]

        mask = mask.to(device=x.device, dtype=torch.bool)

        x = x.reshape(b*n*t,c,h,w)

        cnn_feats = self.feature_extractor(x).flatten(1)

        cnn_seq = cnn_feats.reshape(b * n, t, 2048)  # [B*N, T, 2048]

        player_lstm_out, _ = self.player_lstm(cnn_seq)  # [B*N, T, H]

        # Paper-style concat: x_tk + h_tk
        player_seq = torch.cat([cnn_seq, player_lstm_out], dim=2)
        player_seq = player_seq.reshape(b, n, t, self.player_feat_dim)

        left_feats = player_seq[:, :6]
        right_feats = player_seq[:, 6:]

        left_mask = mask[:, :6]
        right_mask = mask[:, 6:]

        left_pooled = self._masked_group_max(left_feats, left_mask)
        right_pooled = self._masked_group_max(right_feats, right_mask)

        frame_feats = torch.cat([left_pooled, right_pooled], dim=2)
        # [B, T, 2 * (2048 + hidden_size)]

        group_lstm_out, _ = self.frame_lstm(frame_feats)

        logits = self.classifier(group_lstm_out[:, -1])
        return logits