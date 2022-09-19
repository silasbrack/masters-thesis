import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, Subset


class DataModule(ABC, LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        """
        :param data_dir: Where to save/load the data
        :param batch_size: How many samples per batch to load
        :param num_workers: How many workers to use for loading data
        """
        super().__init__()

        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.dataset_train: Union[Dataset, Subset, None] = None
        self.dataset_val: Union[Dataset, Subset, None] = None
        self.dataset_test: Union[Dataset, Subset, None] = None

        # try:
        #     # Note that `datamodule` will be the first dataset to be instantiated.
        #     resolver_name = "datamodule"
        #     OmegaConf.register_new_resolver(resolver_name, lambda name: getattr(self, name), use_cache=False)
        # except ValueError:
        #     logging.debug("This is a workaround for attempting to assign the same resolver twice.")

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_cls(self) -> Dataset:
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def dims(self) -> Tuple[int, int, int]:
    #     raise NotImplementedError

    # @property
    # @abstractmethod
    # def num_classes(self) -> int:
    #     raise NotImplementedError

    @property
    @abstractmethod
    def train_val_test_size(self) -> Tuple[int, int, int]:
        raise NotImplementedError
