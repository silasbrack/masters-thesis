from typing import Tuple

import torch
from torch import Tensor


def get_wrong_indices(preds, y, k) -> Tensor:
    return torch.where(preds.ne(y))[0][:k]


def get_low_conf_indices(conf, k) -> Tuple[Tensor, str]:
    return torch.topk(conf, k=k, dim=0, largest=False).indices


def get_high_conf_indices(conf, k) -> Tuple[Tensor, str]:
    return torch.topk(conf, k=k, dim=0, largest=True).indices


def get_images_and_captions(indices, x, y, preds, conf) -> Tuple[Tensor, str]:
    images = x[indices]
    labels = y[indices].detach().cpu().numpy().tolist()
    preds = preds[indices].detach().cpu().numpy().tolist()
    confs = conf[indices].detach().cpu().numpy().tolist()
    caption = f"Labels = {labels}, " f"Predicted = {preds}, " f"Confidence = {[round(x, 2) for x in confs]}"
    return images, caption
