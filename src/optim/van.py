"""Implementation of Variational Adaptive-Newton algorithm in Pytorch, as described in `Variational
Adaptive-Newton Method for Explorative Learning <https://arxiv.org/abs/1711.05560>`."""
from typing import Iterable

import torch

from src.optim.optimizer import Optimizer


class VariationalAdaptiveNewton(Optimizer):
    """Pytorch optimizer implementation of Variational Adaptive-Newton algorithm."""

    def __init__(self, params: Iterable, lr: float, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.params: Iterable = params
        self.lr: float = lr

    @torch.no_grad()
    def step(self, *args, **kwargs) -> torch.Tensor:
        pass
