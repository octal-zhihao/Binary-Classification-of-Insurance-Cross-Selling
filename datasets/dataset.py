import torch
from torch.utils.data import Dataset

class InsuranceDataset(Dataset):
    def __init__(self, X, y=None):
        # Convert features to float32 tensor
        self.X = torch.tensor(X.values, dtype=torch.float32)
        
        # Convert labels to long tensor if they are not None
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32)  # For regression or binary classification
            # If it's binary classification, you might need to convert it to long
            # self.y = torch.tensor(y.values, dtype=torch.long)
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
