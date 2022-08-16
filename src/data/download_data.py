"""Used to automatically download data to the path under `DATA_PATH` in your `.dotenv` file.

Call using `make data`.
"""
import logging
import os

from dotenv import load_dotenv

from src.data import MNISTDataModule


def download_datasets():
    """Downloads all datasets used in this project to your DATA_PATH environment variable."""
    logger = logging.getLogger(__name__)

    logger.info("Downloading MNIST.")
    MNISTDataModule(os.getenv("DATA_DIR"), 0, 0).prepare_data()


if __name__ == "__main__":
    # pylint: disable = R0801
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    logging.captureWarnings(True)
    load_dotenv()

    download_datasets()
