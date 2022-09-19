import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.functional import accuracy


class MNISTFCNet(nn.Module):
    """Fully-connected neural network for MNIST dataset."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.model(x)


class MNISTFCNetLightningModule(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = MNISTFCNet()
        self.lr = lr
        self.loss = CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        """method used for inference input -> output."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch."""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log("train/loss", loss, sync_dist=True)
        self.log("train/accuracy", acc, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics."""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log("val/loss", loss, sync_dist=True)
        self.log("val/accuracy", acc, sync_dist=True)

        return preds

    def test_step(self, batch, batch_idx):
        """used for logging metrics."""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test/loss", loss, sync_dist=True)
        self.log("test/accuracy", acc, sync_dist=True)

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
