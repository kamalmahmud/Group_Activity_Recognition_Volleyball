import torch
from torch import nn
from models import B5Model


class B5BModel(nn.Module):
    def __init__(self, player_model: B5Model, num_classes=8, freeze_backbone=False):
        super(B5BModel, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.player_model = player_model

        if self.freeze_backbone:
            for param in self.player_model.parameters():
                param.requires_grad = False

        in_feat = self.player_model.fusion_dim
        self.group_classifier = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Linear(in_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, mask=None):
        # x: [B, 12, 9, 3, 224, 224]
        b, n, t, c, h, w = x.shape
        x = x.contiguous().view(b * n, t, c, h, w)

        if self.freeze_backbone:
            self.player_model.eval()
            with torch.no_grad():
                _, player_features = self.player_model(x, return_features=True)
        else:
            _, player_features = self.player_model(x, return_features=True)

        player_features = player_features.reshape(b, n, -1)

        if mask is not None:
            mask = mask.to(device=player_features.device, dtype=torch.bool)
            mask = mask.any(dim=2)
            neg_value = torch.finfo(player_features.dtype).min

            player_features = player_features.masked_fill(
                ~mask.unsqueeze(-1),
                neg_value
            )

        group_features = player_features.max(dim=1).values

        group_logits = self.group_classifier(group_features)

        return group_logits
