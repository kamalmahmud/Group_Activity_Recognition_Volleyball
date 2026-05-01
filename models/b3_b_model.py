import torch
from torch import nn

from models.b3_model import B3AModel


class B3BModel(nn.Module):
    def __init__(self, ckpt_path, num_classes=8):
        super().__init__()
        old_model = B3AModel(num_classes=9)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        old_model.load_state_dict(checkpoint["model_state_dict"])

        # Use ResNet50 as feature extractor
        self.feature_extractor = old_model.model
        # Remove the 9-class classifier and keep 2048 features
        self.feature_extractor.fc = nn.Identity()
        # freeze resnet50
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # x: [batch_size, 12, C, H, W]

        batch_size, num_images, C, H, W = x.shape

        x = x.view(batch_size * num_images, C, H, W)

        features = self.feature_extractor(x)
        # features: [batch_size * 12, 2048]

        features = features.view(batch_size, num_images, 2048)
        # features: [batch_size, 12, 2048]

        pooled_features, _ = torch.max(features, dim=1)
        # pooled_features: [batch_size, 2048]

        output = self.classifier(pooled_features)
        # output: [batch_size, 8]

        return output
