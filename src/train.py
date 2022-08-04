import logging

import hydra
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb_project)

    if cfg.seed:
        torch.manual_seed(cfg.seed)

    data: LightningDataModule = hydra.utils.instantiate(cfg.data)
    data.setup()
    print(next(iter(data.train_dataloader()))[0].shape)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)
    load_dotenv()

    main()
