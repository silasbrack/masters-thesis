from abc import ABC, abstractmethod
from typing import Iterable

import torch


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, params: Iterable, *args, **kwargs):
        pass

    @abstractmethod
    @torch.no_grad()
    def step(self, *args, **kwargs) -> torch.Tensor:
        pass
