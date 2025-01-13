import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchmetrics.functional import accuracy

class CorruptMNISTModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = accuracy(y_pred.softmax(dim=-1), target, task="multiclass", num_classes=10)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
