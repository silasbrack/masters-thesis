import logging
from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class DataModule(ABC, LightningDataModule):
    def __init__(self):
        super().__init__()

        try:
            # Note that `datamodule` will be the first dataset to be instantiated.
            resolver_name = "datamodule"
            OmegaConf.register_new_resolver(resolver_name, lambda name: getattr(self, name), use_cache=False)
        except ValueError:
            logging.debug("This is a workaround for attempting to assign the same resolver twice.")

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
