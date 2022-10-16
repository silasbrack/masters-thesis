import numpy as np
from numpy.random import RandomState

# def data_stream(
#     train_images, train_labels, batch_size: int, drop_last=False, shuffle=True, *, key: jax.random.PRNGKey
# ) -> Tuple[Iterable, Iterable]:
#     dataset_size = len(train_images)
#     num_batches = dataset_size // batch_size if drop_last else dataset_size // batch_size + 1

#     idx = jax.random.permutation(key, dataset_size) if shuffle else jnp.arange(dataset_size)
#     for i in range(num_batches):
#         batch_idx = idx[i * batch_size: (i + 1) * batch_size]
#         yield train_images[batch_idx], train_labels[batch_idx]

def data_stream(images, labels, batch_size, key):
    num_train = len(images)
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    rng = np.random.RandomState(key)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1) * batch_size]
            yield images[batch_idx], labels[batch_idx]