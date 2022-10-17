import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from matplotlib import animation
from omegaconf import DictConfig, OmegaConf

import wandb


def get_run(run_path: str):
    run_id = json.load(open(f"{run_path}/.wandb.json", "r"))["run_id"]
    api = wandb.Api()
    entity, project = os.getenv("WANDB_ENTITY"), os.getenv("WANDB_PROJECT")
    run = api.run(entity + "/" + project + "/" + run_id)
    return run


def get_run_data(run) -> pd.DataFrame:
    history = run.scan_history()
    df = pd.DataFrame(history)
    return df


def plot_loss_curves(df: pd.DataFrame, ax: plt.Axes):
    pdf_epoch = df[["trainer/global_step", "epoch"]].groupby("epoch").nth(0).reset_index()
    pdf_epoch = pdf_epoch.loc[range(0, len(pdf_epoch), 2)]

    (line,) = ax.plot(df["trainer/global_step"], df["train/loss"])
    ax.plot(df["trainer/global_step"], df["val/loss"], marker="o", color=line.get_color())
    ax.set_xticks(pdf_epoch["trainer/global_step"], pdf_epoch["epoch"])


def format_number(data_value, indx):
    if data_value >= 1_000_000:
        formatter = "{:1.1f}M".format(data_value * 0.000_001)
    elif data_value >= 1_000:
        formatter = "{:1.0f}k".format(data_value * 0.001)
    else:
        formatter = "{:1.0f}".format(data_value)
    return formatter


def get_splits(path: str):
    conf_path = f"{path}/.hydra/config.yaml"
    conf = OmegaConf.load(conf_path)
    model = instantiate(conf.model)
    params_per_layer = [sum(torch.numel(p) for p in layer.parameters()) for layer in model.model]
    params_per_layer = list(filter(lambda x: x > 0, params_per_layer))
    num_layers = len(params_per_layer)
    param_limits = [0] + np.cumsum(params_per_layer).tolist()
    param_limits = (np.array(param_limits) - 0.5).tolist()
    return param_limits, num_layers


def get_min_max_value(run_paths: List[str], n_epochs: int):
    vmin = np.inf
    vmax = -np.inf
    for run_path in run_paths:
        for e in range(n_epochs):
            posterior_precision = torch.load(
                f"{run_path}/posterior_precision_epoch{e}_opt_diag.pt", map_location="cpu"
            ).numpy()
            vmin = min(vmin, posterior_precision.min())
            vmax = max(vmax, posterior_precision.max())
            posterior_precision = torch.load(
                f"{run_path}/posterior_precision_epoch{e}_unopt_diag.pt", map_location="cpu"
            ).numpy()
            vmin = min(vmin, posterior_precision.min())
            vmax = max(vmax, posterior_precision.max())

    vmin = min(vmin, 0)
    return vmin, vmax


def visualize_hessians_diag(run_paths: List[str]):
    splits_layers = [get_splits(path) for path in run_paths]
    splits: List[List[int]] = [s[0] for s in splits_layers]
    num_layers: List[int] = [s[1] for s in splits_layers]

    fig, ax = plt.subplots(nrows=5, ncols=2, sharey="row", figsize=(14, 10))
    epoch = 2
    struct = "diag"
    for i, (num_layer, run_path, split) in enumerate(zip(num_layers, run_paths, splits)):
        posterior_precision_opt = torch.load(
            f"{run_path}/posterior_precision_epoch{epoch}_opt_{struct}.pt", map_location="cpu"
        ).numpy()
        ax[i, 0].plot(posterior_precision_opt)
        ax[i, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x, pos)))
        for i, s in enumerate(splits):
            ax[i, 0].vlines(x=s, ymin=0, ymax=max(posterior_precision_opt), colors="slategrey", linestyles="--")

        posterior_precision_unopt = torch.load(
            f"{run_path}/posterior_precision_epoch{epoch}_unopt_{struct}.pt", map_location="cpu"
        ).numpy()
        ax[i, 1].plot(posterior_precision_unopt)
        ax[i, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x, pos)))
        for i, s in enumerate(splits):
            ax[i, 1].vlines(x=s, ymin=0, ymax=max(posterior_precision_unopt), colors="slategrey", linestyles="--")

        ax[i, 0].set_title(f"{i+1} layers, optimized")
        ax[i, 1].set_title(f"{i+1} layers, unoptimized")
        ax[i, 0].set_ylabel("Posterior precision")
    ax[-1, 0].set_xlabel("Parameter index")
    fig.tight_layout()
    fig.savefig("hessian_diag.png", dpi=300)


def create_animation(run_paths: List[str], n_epochs: int, struct: str):
    splits_layers = [get_splits(path) for path in run_paths]
    splits: List[List[int]] = [s[0] for s in splits_layers]
    num_layers: List[int] = [s[1] for s in splits_layers]

    fig, axs = plt.subplots(nrows=len(run_paths), ncols=3, figsize=(15, 10))
    vmin, vmax = get_min_max_value(run_paths, n_epochs)

    def init_plots(axs):
        lines = []
        for i, ax in enumerate(axs):
            ax_opt = ax[0]
            ax_unopt = ax[1]
            ax_loss = ax[2]
            run = get_run(run_paths[i])
            # num_layers = json.loads(run.json_config)["num_layers"]["value"]
            df = get_run_data(run)

            plot_loss_curves(df, ax_loss)
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")

            lines.append(ax_opt.plot([], [])[0])
            ax_opt.set_title(f"{num_layers[i]} layers, optimized")
            ax_opt.set_ylabel("Posterior precision")
            ax_opt.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x, pos)))
            ax_opt.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x, pos)))

            lines.append(ax_unopt.plot([], [])[0])
            ax_unopt.set_title(f"{num_layers[i]} layers, unoptimized")
            ax_unopt.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x, pos)))
            ax_unopt.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x, pos)))

            for j, s in enumerate(splits[i]):
                ax_opt.vlines(x=s, ymin=vmin, ymax=vmax, colors="#b0b0b0", linestyles="-", linewidth=0.8)
                ax_unopt.vlines(x=s, ymin=vmin, ymax=vmax, colors="#b0b0b0", linestyles="-", linewidth=0.8)

            ax_opt.set_xlim(min(splits[i]), max(splits[i]))
            ax_unopt.set_xlim(min(splits[i]), max(splits[i]))
            ax_opt.set_ylim(vmin, vmax)
            ax_unopt.set_ylim(vmin, vmax)
            ax_opt.grid(False)
            ax_unopt.grid(False)

        axs[-1][0].set_xlabel("Parameter index")
        axs[-1][1].set_xlabel("Parameter index")
        return lines

    def init():
        [line.set_data([], []) for line in lines]
        return lines

    def animate(i):
        for j in range(len(run_paths)):

            posterior_precision = torch.load(
                f"{run_paths[j]}/posterior_precision_epoch{i}_opt_{struct}.pt", map_location="cpu"
            ).numpy()
            lines[2 * j].set_data(np.arange(posterior_precision.shape[0]), posterior_precision)

            posterior_precision = torch.load(
                f"{run_paths[j]}/posterior_precision_epoch{i}_unopt_{struct}.pt", map_location="cpu"
            ).numpy()
            lines[2 * j + 1].set_data(np.arange(posterior_precision.shape[0]), posterior_precision)

        return lines

    lines = init_plots(axs)
    fig.suptitle(f"Hessian structure: {struct}")
    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_epochs, interval=1000, blit=True)
    anim.save("results/posterior_precision_diag.mp4", writer="ffmpeg", fps=1)


if __name__ == "__main__":
    load_dotenv()

    # run_paths = [
    #     "logs/multiruns/2022-09-19_22-09-53/0",
    #     "logs/multiruns/2022-09-19_22-09-53/1",
    #     "logs/multiruns/2022-09-19_22-09-53/2",
    #     "logs/multiruns/2022-09-19_22-09-53/3",
    #     "logs/multiruns/2022-09-19_22-09-53/4",
    # ]
    run_paths = [
        "logs/multiruns/2022-09-21_12-32-25/0",
        "logs/multiruns/2022-09-21_12-32-25/1",
        "logs/multiruns/2022-09-21_12-32-25/2",
        "logs/multiruns/2022-09-21_12-32-25/3",
        "logs/multiruns/2022-09-21_12-32-25/4",
        "logs/multiruns/2022-09-21_12-32-25/5",
        "logs/multiruns/2022-09-21_12-32-25/6",
    ]

    create_animation(run_paths, n_epochs=10, struct="diag")
