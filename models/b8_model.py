import torch
import torch.nn as nn

from models.b5_model import B5Model

class B8Model(nn.Module):
    def __init__(self,num_classes=8,hidden_size=512,player_feature_size=2048):
        super(B8Model, self).__init__()
        self.player_feature_dim = player_feature_size + hidden_size
        self.player_model = B5Model(hidden_size=hidden_size)

        self.frame_projection = nn.Sequential(
            nn.LayerNorm(2 * self.player_feature_dim),
            nn.Linear(2 * self.player_feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.frame_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self,x,mask=None):
        b,n,t,c,h,w = x.shape
        x = x.reshape(b*n,t,c,h,w)

        player_out = self.player_model(x,return_all_steps=True)
        # [B*N, T, 2048 + hidden_size]

        player_out = player_out.reshape(b, n, t, -1)
        # [B, N, T, 2048 + hidden_size]

        if mask is not None:
            mask = mask.to(device=player_out.device, dtype=torch.bool)
            neg_val = torch.finfo(player_out.dtype).min
            player_out = player_out.masked_fill(~mask.unsqueeze(-1), neg_val)

        left_group = player_out[:,:6,:,:]
        right_group = player_out[:,6:,:,:]

        left_group = left_group.max(dim=1).values
        right_group = right_group.max(dim=1).values
        # both: [B, T, 2048 + hidden_size]

        frame_feats = torch.cat((left_group, right_group), dim=2)
        # [B, T, 2 * (2048 + hidden_size)]

        frame_feats = self.frame_projection(frame_feats)
        # [B, T, hidden_size]

        lstm_out,_ = self.frame_lstm(frame_feats)
        # [B, T, hidden_size]

        out = lstm_out[:,-1,:]
        # [B, hidden_size]

        out = self.classifier(out)
        # [B, num_classes]
        
        return out