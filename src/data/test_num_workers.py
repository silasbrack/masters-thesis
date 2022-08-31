import os
from time import time
import multiprocessing as mp

from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.mnist import MNISTDataModule


load_dotenv()

dm = MNISTDataModule(os.getenv("DATA_DIR"), batch_size=64, num_workers=0)
dm.setup()

for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = DataLoader(dm.dataset_train, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
