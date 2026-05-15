import torch
import torch.nn as nn
from models.b5_model import B5Model

class B8Model(nn.Module):
    def __init__(self,num_classes=8,hidden_size=512):
        super(B8Model, self).__init__()
        self.player_model = B5Model()
        player_feature_dim = self.player_model.fusion_dim  # 2048 + hidden_size*2

        # Independent projections — each team: 4096 → 1024
        self.left_projection = nn.Sequential(
            nn.LayerNorm(player_feature_dim),
            nn.Linear(player_feature_dim, hidden_size * 2),  # 4096 → 1024
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.right_projection = nn.Sequential(
            nn.LayerNorm(player_feature_dim),
            nn.Linear(player_feature_dim, hidden_size * 2),  # 4096 → 1024
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.frame_lstm = nn.LSTM(
            input_size=hidden_size * 4,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self,x,mask=None):
        b,n,t,c,h,w = x.shape
        x = x.reshape(b*n,t,c,h,w)

        player_out = self.player_model(x,return_all_steps=True) # [B*N, T, fusion_dim]

        player_out = player_out.reshape(b, n, t, -1) # [B, N, T, fusion_dim]

        if mask is not None:
            mask = mask.to(device=player_out.device, dtype=torch.bool)
            neg_val = torch.finfo(player_out.dtype).min
            player_out = player_out.masked_fill(~mask.unsqueeze(-1), neg_val)

        left_group = player_out[:,:6,:,:].max(dim=1).values
        right_group = player_out[:,6:,:,:].max(dim=1).values
        # both: [B, T, 4096]

        left_group  = self.left_projection(left_group)            # [B, T, 1024]
        right_group = self.right_projection(right_group)           # [B, T, 1024]

        frame_feats = torch.cat((left_group, right_group), dim=2)  # [B, T, 1024*2]

        lstm_out,_ = self.frame_lstm(frame_feats) # [B, T, 1024]

        out = self.classifier(lstm_out[:,4,:])
        # [B, num_classes]
        
        return out