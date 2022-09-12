from typing import Iterable, Tuple

import jax
from jax import numpy as jnp


def data_stream(
    train_images, train_labels, batch_size: int, drop_last=False, shuffle=True, *, key: jax.random.PRNGKey
) -> Tuple[Iterable, Iterable]:
    dataset_size = len(train_images)
    num_batches = dataset_size // batch_size if drop_last else dataset_size // batch_size + 1

    idx = jax.random.permutation(key, dataset_size) if shuffle else jnp.arange(dataset_size)
    for i in range(num_batches):
        batch_idx = idx[i * batch_size: (i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
