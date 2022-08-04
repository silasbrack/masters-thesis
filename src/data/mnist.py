import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    def __init__(self, dir: str, batch_size: int, num_workers: int):
        super().__init__()

        self.name: str = "MNIST"
        self.size: int = 60000
        self.data_dir: str = dir
        self.batch_size: int = batch_size
        self.eval_batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.n_classes: int = 10
        self.resolution: int = 28
        self.channels: int = 1

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
        )
