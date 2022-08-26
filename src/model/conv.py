import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.functional import accuracy


class MNISTConvNet(LightningModule):
    def __init__(self, lr, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 12, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 12, n_classes),
        )
        self.lr = lr
        self.n_classes = n_classes
        self.loss = CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        """method used for inference input -> output."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch."""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log("train/loss", loss)
        self.log("train/accuracy", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics."""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log("val/loss", loss)
        self.log("val/accuracy", acc)

        return preds

    def test_step(self, batch, batch_idx):
        """used for logging metrics."""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test/loss", loss)
        self.log("test/accuracy", acc)

    def configure_optimizers(self):
        """defines model optimizer."""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar."""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
