import torch
import torch.nn.functional as F
from torch import nn


class InsuranceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(InsuranceModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))
        return x
