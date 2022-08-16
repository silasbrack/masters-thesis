"""Trains a neural network and evaluates its performance."""
import logging

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

import wandb
from src.data.data_module import DataModule
from src.optim.optimizer import Optimizer
from src.utils import (
    get_high_conf_indices,
    get_images_and_captions,
    get_low_conf_indices,
    get_wrong_indices,
)


@hydra.main(config_path="../conf/", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Runs training procedure.

    :param cfg: DictConfig, Hydra config, automatically obtained from ./conf/config.yaml
    """
    wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    if cfg.seed:
        torch.manual_seed(cfg.seed)

    data: DataModule = hydra.utils.instantiate(cfg.data)
    data.setup()

    model: nn.Module = hydra.utils.instantiate(cfg.models, n_classes=data.num_classes)
    wandb.watch(model)
    optim: Optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = CrossEntropyLoss()
    accuracy_fn = Accuracy()

    k: int = cfg.n_images_to_visualize  # Number of images to visualize

    first_batch_x, first_batch_y = next(iter(data.train_dataloader()))
    wandb.log({"input": wandb.Image(first_batch_x[:k], caption=f"Labels = {first_batch_y[:k].numpy().tolist()}")})

    for epoch in range(cfg.epochs):
        accuracy_fn.reset()
        for x, y in data.train_dataloader():
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            accuracy = accuracy_fn(logits, y)
            loss.backward()
            optim.step()
            wandb.log({"epoch": epoch + 1, "train/loss": loss.detach().cpu().item(), "train/accuracy": accuracy})
    wandb.log({"train/final_accuracy": accuracy_fn.compute()})

    confs = []
    predictions = []
    labels = []
    for i, (x, y) in enumerate(data.test_dataloader()):
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        logits: Tensor = model(x)
        probs: Tensor = torch.softmax(logits, dim=-1)
        conf, preds = torch.max(probs, dim=-1)
        accuracy_fn(logits, y)

        confs.append(conf)
        predictions.append(preds)
        labels.append(y)

        if i == 0:
            wrong_indices = get_wrong_indices(preds, y, k)
            wrong_images, wrong_caption = get_images_and_captions(wrong_indices, x, y, preds, conf)
            wandb.log({"wrong": wandb.Image(wrong_images, caption=wrong_caption)})

            high_conf_indices = get_high_conf_indices(conf, k)
            high_conf_images, high_conf_caption = get_images_and_captions(high_conf_indices, x, y, preds, conf)
            wandb.log({"high_confidence": wandb.Image(high_conf_images, caption=high_conf_caption)})

            low_conf_indices = get_low_conf_indices(conf, k)
            low_conf_images, low_conf_caption = get_images_and_captions(low_conf_indices, x, y, preds, conf)
            wandb.log({"low_confidence": wandb.Image(low_conf_images, caption=low_conf_caption)})

    confs = torch.cat(confs, dim=0)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    wandb.log(
        {
            "test/final_accuracy": accuracy_fn.compute(),
            "test/confidence": confs,
            "test/predictions": predictions,
            "test/labels": labels,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    # pylint: disable = R0801
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    logging.captureWarnings(True)
    load_dotenv()

    main()  # pylint: disable = no-value-for-parameter
