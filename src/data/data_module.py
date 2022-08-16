from abc import ABC, abstractmethod
from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class DataModule(ABC, LightningDataModule):
    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError
