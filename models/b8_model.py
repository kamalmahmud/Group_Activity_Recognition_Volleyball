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

        self.frame_lstm = nn.LSTM(
            input_size=hidden_size * 2,
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

    def forward(self,x,mask=None):
        # batch, players, time, channels, height, width
        b,n,t,c,h,w = x.shape
        x = x.reshape(b*n*t,c,h,w)

        player_feats = self.feature_extractor(x)
        player_feats = player_feats.flatten(1)# [b*n*t,2048]

        player_seq = player_feats.reshape(b * n, t, -1) # [B*N, T, 2048]

        player_lstm_out, _ = self.player_lstm(player_seq) # [b*n, t, hidden_size]

        players = player_lstm_out.reshape(b, n, t, -1) # [b, n, t, hidden_size]

        if mask is not None:
            # mask: [b, n, t], True = real player, False = padded player
            mask = mask.to(players.device).bool()

            left_mask = mask[:, :6, :].unsqueeze(-1)
            right_mask = mask[:, 6:, :].unsqueeze(-1)

            neg_value = -torch.finfo(players.dtype).max

            left_feats = players[:, :6, :, :].masked_fill(~left_mask, neg_value)
            right_feats = players[:, 6:, :, :].masked_fill(~right_mask, neg_value)

            left_feats = left_feats.max(dim=1)[0]
            right_feats = right_feats.max(dim=1)[0]
        else:
            left_feats = players[:, :6, :, :].max(dim=1)[0]
            right_feats = players[:, 6:, :, :].max(dim=1)[0]

        frame_feats = torch.cat((left_feats, right_feats), dim=2)  # [B, T, hidden_size*2]

        lstm_out,_ = self.frame_lstm(frame_feats) # [B, T, hidden_size]

        out = self.classifier(lstm_out[:, -1, :]) # [b, num_classes]
        
        return out