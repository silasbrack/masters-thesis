import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.functional import accuracy


class MNISTConvNet(nn.Module):
    """Convolutional neural network for MNIST dataset."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=-1),
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 12, 3, stride=1, padding=1, bias=False),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(12, 12, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(7 * 7 * 12, 10),
        #     nn.LogSoftmax(dim=-1),
        # )

    def forward(self, x):
        return self.model(x)


class MNISTConvNetLightningModule(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = MNISTConvNet()
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
