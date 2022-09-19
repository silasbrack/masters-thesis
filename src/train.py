"""Trains a neural network and evaluates its performance."""
import json
import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.data.data_module import DataModule


@hydra.main(config_path="../conf/", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Runs training procedure.

    :param cfg: DictConfig, Hydra config, automatically obtained from ./conf/config.yaml
    """
    if cfg.seed:
        seed_everything(cfg.seed)

    wandb.init(reinit=True)
    with open(".wandb.json", "w") as f:
        json.dump({"run_id": wandb.run.id}, f)

    data: DataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=WandbLogger(),
        # callbacks=[
        #     ModelCheckpoint(dirpath="checkpoints/", monitor="val/accuracy", mode="max", save_last=True),
        # ],
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
    wandb.finish()


if __name__ == "__main__":
    logging.captureWarnings(True)
    load_dotenv()

    main()  # pylint: disable = no-value-for-parameter
