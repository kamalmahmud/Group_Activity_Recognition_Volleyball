import torch
from torch import nn
from models import B5Model


class B5BModel(nn.Module):
    def __init__(self, player_model: B5Model, hidden_size=512, num_classes=8, freeze_backbone=False):
        super(B5BModel, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.player_model = player_model

        if self.freeze_backbone:
            for param in self.player_model.parameters():
                param.requires_grad = False

        self.group_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
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

        group_features, _ = player_features.max(dim=1)

        group_logits = self.group_classifier(group_features)

        return group_logits
