"""Abstract class for Bayesian optimizers implemented in this project."""
from abc import ABC, abstractmethod
from typing import Iterable

import torch


class Optimizer(ABC, torch.optim.Optimizer):
    """Optimizer class for Bayesian optimizer.

    Places a probability distribution on the weights which it optimizes.
    """

    @abstractmethod
    def __init__(self, params: Iterable, **kwargs):
        super().__init__(params, kwargs)

    @abstractmethod
    @torch.no_grad()
    def step(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
