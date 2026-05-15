import torch
from torch import nn

from models.b3_a_model import B3AModel


class B6Model(nn.Module):
    def __init__(
        self,
        ckpt_path,
        num_classes=8,
        num_person_classes=9,
        cnn_feature_size=2048,
        lstm_hidden_size=2000,
        lstm_num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        b3a_model = B3AModel(num_classes=num_person_classes)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        # Handle DataParallel checkpoints if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        # B6 does not need the B3A classifier head.
        # Ignore all model.fc.* weights because your checkpoint head differs.
        backbone_state_dict = {
            k: v for k, v in new_state_dict.items()
            if not k.startswith("model.fc.")
        }

        missing, unexpected = b3a_model.load_state_dict(
            backbone_state_dict,
            strict=False
        )

        self.feature_extractor = b3a_model.model

        self.feature_extractor.fc = nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,       # 2048
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x, valid_mask=None):
       # x shape: [B, T, N, C, H, W]
        b, t, n, c, h, w = x.shape

        if valid_mask is None:
            valid_mask = x.flatten(3).abs().sum(dim=3) > 0
            # [B, T, N]

        x = x.reshape(b * t * n, c, h, w)

        with torch.no_grad():
            self.feature_extractor.eval()
            features = self.feature_extractor(x)
            # [B*T*N, 2048]

        features = features.reshape(b, t, n, -1)
        # [B, T, N, 2048]

        valid_mask = valid_mask.unsqueeze(-1)
        # [B, T, N, 1]

        # Ignore padded crops during max pooling
        mask_value = torch.finfo(features.dtype).min
        features = features.masked_fill(~valid_mask, mask_value)

        frame_features = torch.max(features, dim=2).values
        # [B, T, 2048]

        any_valid = valid_mask.any(dim=2)
        # [B, T, 1]

        frame_features = torch.where(
            any_valid,
            frame_features,
            torch.zeros_like(frame_features),
        )

        lstm_out, _ = self.lstm(frame_features)
        # [B, T, lstm_hidden_size]

        clip_features = lstm_out[:, -1, :]
        # [B, lstm_hidden_size]

        logits = self.classifier(clip_features)
        # [B, 8]

        return logits