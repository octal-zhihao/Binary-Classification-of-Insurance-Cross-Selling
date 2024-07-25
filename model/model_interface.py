from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torchmetrics
from .model import InsuranceModel

class MInterface(LightningModule):
    def __init__(self, input_dim, lr, num_heads, dropout_rate=0.5):
        super(MInterface, self).__init__()
        self.model = InsuranceModel(input_dim, num_heads, dropout_rate)
        self.criterion = nn.MSELoss()
        self.lr = lr

        self.f1_score = torchmetrics.F1Score(task='binary', threshold=0.5)
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        preds = torch.round(y_hat)
        f1 = self.f1_score(preds, y.unsqueeze(1))
        acc = self.accuracy(preds, y.unsqueeze(1))
        precision = self.precision(preds, y.unsqueeze(1))
        recall = self.recall(preds, y.unsqueeze(1))

        self.log('train_acc', acc)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        preds = torch.round(y_hat)
        f1 = self.f1_score(preds, y.unsqueeze(1))
        acc = self.accuracy(preds, y.unsqueeze(1))
        precision = self.precision(preds, y.unsqueeze(1))
        recall = self.recall(preds, y.unsqueeze(1))

        self.log('val_acc', acc)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer