import torch
from torch.utils.data import Dataset

class InsuranceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        return x
