import torch
import torch.nn.functional as F
from torch import nn

class InsuranceModel(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate=0.5):
        super(InsuranceModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        # 注意力机制
        x = x.unsqueeze(1)  # (batch_size, seq_len, embed_dim)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)  # (batch_size, embed_dim)

        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer_3(x))
        return x