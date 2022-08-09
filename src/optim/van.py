"""Implementation of Variational Adaptive-Newton algorithm in Pytorch, as described in `Variational
Adaptive-Newton Method for Explorative Learning <https://arxiv.org/abs/1711.05560>`."""
from typing import Iterable

import torch

from src.optim.optimizer import Optimizer


class VariationalAdaptiveNewton(Optimizer):
    def __init__(self, params: Iterable, lr: float):
        self.params: Iterable = params
        self.lr: float = lr

    def step(self) -> torch.Tensor:
        pass
