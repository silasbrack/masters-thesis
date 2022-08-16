"""Contains MNIST data module.

Typical usage example:

dm = MNISTDataModule("data/", batch_size=64, num_workers=4)
dm.setup()
train_loader = dm.train_dataloader()
"""
from typing import Union

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from src.data.data_module import DataModule


class MNISTDataModule(DataModule):
    """
    .. figure:: https://miro.medium.com/max/744/1*AO2rIhzRYzFVQlFLx9DM9A.png
        :width: 400
        :alt: MNIST
    Specs:
        - 10 classes (1 per digit)
        - Each image is (1 x 28 x 28)
    Standard MNIST, train, val, test splits and transforms.
    Transforms::
        mnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    Example::
        from pl_bolts.datamodules import MNISTDataModule
        dm = MNISTDataModule('.')
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "MNIST"
    dataset_cls = MNIST
    dims = (1, 28, 28)

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

        self.dataset_train: Union[Dataset, None] = None
        self.dataset_val: Union[Dataset, None] = None
        self.dataset_test: Union[Dataset, None] = None

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # Download MNIST data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10
