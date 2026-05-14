import torch
import torch.nn as nn

from models.b5_model import B5Model


class B6Model(nn.Module):
    def __init__(self, ckpt_path: str, num_classes: int = 8, hidden_size: int = 512, num_layers: int = 1,
                 freeze_backbone: bool = False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        b5 = B5Model()

        self.feature_extractor = b5.model
        if self.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        #   b,n,t,c,h,w
        b, n, t, c, h, w = x.shape
        # reshape to extract features
        x = x.reshape(b * n * t, c, h, w)

        if self.freeze_backbone:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
        # b,n,t,2048
        features = features.reshape(b, n, t, -1)

        if mask is not None:
            mask = mask.to(device=features.device, dtype=torch.bool)
            neg_val = torch.finfo(features.dtype).min
            features = features.masked_fill(~mask.unsqueeze(-1), neg_val)

        # max pooling over players -> b,t,2048
        frame_feats = features.max(dim=1).values

        lstm_out, _ = self.lstm(frame_feats)  # b,t,hidden_size

        output = lstm_out[:, -1, :]  # b,hidden_size
        output = self.classifier(output)  # b, num_classes

        return output
