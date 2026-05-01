import torch
import torch.nn as nn

from models.b3_model import B3AModel


class B3BModel(nn.Module):
    def __init__(self, ckpt_path, num_classes=8):
        super().__init__()
        old_model = B3AModel(num_classes=9)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        old_model.load_state_dict(checkpoint["model_state_dict"])

        self.feature_extractor = old_model.model

        # remove Dropout + Linear(2048 -> 9)
        self.feature_extractor.fc = nn.Identity()
        # freeze ResNet50
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, 12, C, H, W]

        B, N, C, H, W = x.shape

        # detect zero-padded crops
        valid_mask = x.flatten(2).abs().sum(dim=2) > 0
        # shape: [B, 12]

        x = x.reshape(B * N, C, H, W)

        with torch.no_grad():
            features = self.feature_extractor(x)
            # shape: [B * 12, 2048]

        features = features.reshape(B, N, 2048)
        # shape: [B, 12, 2048]

        valid_mask = valid_mask.unsqueeze(-1)
        # shape: [B, 12, 1]

        # ignore padded crops during max pooling
        features = features.masked_fill(~valid_mask, -1e9)

        left_features = features[:, :6, :]
        right_features = features[:, 6:, :]

        left_valid = valid_mask[:, :6, :]
        right_valid = valid_mask[:, 6:, :]

        left_pooled = torch.max(left_features, dim=1).values
        right_pooled = torch.max(right_features, dim=1).values

        # if a side has no valid persons, replace -1e9 with zeros
        left_any = left_valid.any(dim=1)
        right_any = right_valid.any(dim=1)

        left_pooled = torch.where(
            left_any,
            left_pooled,
            torch.zeros_like(left_pooled)
        )

        right_pooled = torch.where(
            right_any,
            right_pooled,
            torch.zeros_like(right_pooled)
        )

        pooled = torch.cat([left_pooled, right_pooled], dim=1)
        # shape: [B, 4096]

        output = self.classifier(pooled)
        # shape: [B, 8]

        return output
