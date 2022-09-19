"""Contains MNIST data module.

Typical usage example:

dm = HousingDataModule("data/", batch_size=64, num_workers=4)
dm.setup()
train_loader = dm.train_dataloader()
"""
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

from src.data.data_module import DataModule


class HousingDataset(torch.utils.data.Dataset):
    """Prepare the Boston dataset for regression."""

    def __init__(self, scale_data=True):
        X, y = fetch_california_housing(return_X_y=True)
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class HousingDataModule(DataModule):
    name = "Housing"
    dataset_cls = HousingDataset
    num_features = 13
    train_val_test_size = (15640, 2000, 3000)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = HousingDataset()
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(dataset, self.train_val_test_size)
