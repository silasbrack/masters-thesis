import logging

from dotenv import load_dotenv

from src.data import MNISTDataModule


def download_datasets():
    logger = logging.getLogger(__name__)

    logger.info("Downloading MNIST.")
    MNISTDataModule(0, 0).prepare_data()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)
    load_dotenv()

    download_datasets()
