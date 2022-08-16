"""Implementation of Noisy Natural Gradient algorithm in Pytorch, as described in `Noisy Natural
Gradient as Variational Inference <http://proceedings.mlr.press/v80/zhang18l/zhang18l.pdf>`."""
from typing import Iterable

import torch

from src.optim.optimizer import Optimizer


class NoisyNaturalGradient(Optimizer):
    """Pytorch optimizer implementation of Noisy Natural Gradient algorithm."""

    def __init__(self, params: Iterable, lr: float, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.params: Iterable = params
        self.lr: float = lr

    @torch.no_grad()
    def step(self, *args, **kwargs) -> torch.Tensor:
        pass
