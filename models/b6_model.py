import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

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


        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.feature_extractor.fc = nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,       # 2048
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x,mask=None):
       # x shape: [B, N, T, C, H, W]
        b, n, t, c, h, w = x.shape

        if mask is None:
            mask = x.flatten(3).abs().sum(dim=3) > 0
            # [B, t, N]

        x = x.reshape(b * n * t, c, h, w)

        with torch.no_grad():
            self.feature_extractor.eval()
            features = self.feature_extractor(x)
            # [B*t*N, 2048]

        features = features.reshape(b, n, t, -1)
        # [B, N, T, 2048]

        mask = mask.unsqueeze(-1)
        # [B, N, T, 1]

        # Ignore padded crops during max pooling
        mask_value = torch.finfo(features.dtype).min
        features = features.masked_fill(~mask, mask_value)

        frame_features = torch.max(features, dim=1).values
        # [B, t, 2048]

        any_valid = mask.any(dim=1)
        # [B, t, 1]

        frame_features = torch.where(
            any_valid,
            frame_features,
            torch.zeros_like(frame_features),
        )

        lstm_out, _ = self.lstm(frame_features)
        # [B, t, lstm_hidden_size]

        clip_features = lstm_out[:, -1, :]
        # [B, lstm_hidden_size]

        logits = self.classifier(clip_features)
        # [B, 8]

        return logits