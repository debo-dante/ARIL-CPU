import torch
import torch.nn as nn

class TransformerCSI(nn.Module):
    def __init__(self, input_dim=30, seq_len=3000, hidden_dim=128, num_heads=4, num_layers=3, num_classes_act=6, num_classes_loc=16):
        super(TransformerCSI, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_act = nn.Linear(hidden_dim, num_classes_act)
        self.fc_loc = nn.Linear(hidden_dim, num_classes_loc)

    def forward(self, x):  # x: [batch, channels, time]
        x = x.permute(0, 2, 1)  # [batch, time, channels]
        x = self.embedding(x)   # [batch, time, hidden_dim]
        x = self.transformer(x)
        x = x.mean(dim=1)       # global average pooling over time
        act = self.fc_act(x)
        loc = self.fc_loc(x)
        return act, loc
