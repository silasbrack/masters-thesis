"""Module containing CNNs for MNIST."""
import torch
from torch import nn


class MNISTConvNet(nn.Module):
    """CNN for MNIST."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
