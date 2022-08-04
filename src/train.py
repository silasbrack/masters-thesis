import logging

import hydra
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project="masters-thesis")

    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data: LightningDataModule = hydra.utils.instantiate(cfg.data)
    data.setup()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)
    load_dotenv()

    main()
