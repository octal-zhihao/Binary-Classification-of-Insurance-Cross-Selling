import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .model import InsuranceModel


class MInterface(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.model = InsuranceModel(input_dim, hidden_dim)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
