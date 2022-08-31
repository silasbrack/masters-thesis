"""Contains MNIST data module.

Typical usage example:

dm = MNISTDataModule("data/", batch_size=64, num_workers=4)
dm.setup()
train_loader = dm.train_dataloader()
"""

from torch.utils.data import random_split
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
        dm = MNISTDataModule("data/", batch_size=64, num_workers=4)
        dm.setup()
        train_loader = dm.train_dataloader()
    """

    name = "MNIST"
    dataset_cls = MNIST
    dims = (1, 28, 28)
    num_classes = 10
    train_val_test_size = (55000, 5000, 10000)

    transform = transforms.Compose(
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
