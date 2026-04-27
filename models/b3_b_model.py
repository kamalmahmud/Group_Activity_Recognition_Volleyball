from torch import nn


class B3Model(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
