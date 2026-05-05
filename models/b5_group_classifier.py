from torch import nn

from models import B5Model


class B5BModel(nn.Module):
    def __init__(self, player_model: B5Model, hidden_size=512, num_classes=8, freeze_backbone=False):
        super(B5BModel, self).__init__()
        self.player_model = player_model

        if freeze_backbone:
            for param in self.player_model.parameters():
                param.requires_grad = False

        self.group_classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
                x shape:
                [batch_size, num_players, num_frames, channels, height, width]

                Example:
                [B, 12, T, 3, H, W]
                """
        b, n, t, c, h, w = x.shape
        x = x.reshape(b * n, t, c, h, w)

        _, player_features = self.player_model(x, return_features=True)
        # [batch size * num_players, hidden_size]
        player_features = player_features.reshape(b, n, -1)

        # max pooling over players
        group_features, _ = player_features.max(dim=1)

        group_logits = self.group_classifier(group_features)

        return group_logits
