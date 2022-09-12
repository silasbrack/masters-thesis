from functools import partial
import logging
import os
import time

from dotenv import load_dotenv
import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from numpy.random import RandomState
import wandb
from src_jax.data.dataloader import data_stream

from src_jax.data.mnist import mnist
from src_jax.model import FcNet, ConvNet
from src_jax.utils import compute_loader_accuracy, compute_accuracy


# @eqx.filter_value_and_grad
def compute_loss(model, inputs, targets):
    preds = jax.vmap(model, in_axes=0)(inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))

if __name__ == "__main__":
    load_dotenv()
    wandb.init(project=os.getenv("WANDB_PROJECT"))

    data_dir = os.getenv("DATA_DIR")

    rng = random.PRNGKey(0)

    # model = FcNet(rng)
    model = ConvNet(rng)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model))
    print(f"Number of parameters: {num_params}")

    step_size = 0.001
    num_epochs = 10
    batch_size = 512

    train_images, train_labels, test_images, test_labels = mnist(data_dir)
    train_images = jnp.array(train_images)
    train_labels = jnp.array(train_labels)
    test_images = jnp.array(test_images)
    test_labels = jnp.array(test_labels)
    num_train = len(train_images)

    # dataset_train = MNIST("/tmp/mnist/", train=True, download=True, transform=FlattenAndCast(),
    #                       target_transform=FlattenAndCast())
    # dataset_test = MNIST("/tmp/mnist/", train=False, download=True, transform=FlattenAndCast(),
    #                      target_transform=FlattenAndCast())
    # train_loader = NumpyLoader(dataset_train, batch_size=batch_size, num_workers=0)
    # test_loader = NumpyLoader(dataset_test, batch_size=batch_size, num_workers=0)
    # num_train = len(dataset_train)

    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    batches = data_stream(train_images, train_labels, batch_size)
    optim = optax.adam(step_size)
    opt_state = optim.init(model)

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
        # loss, grads = compute_loss(model, x, y)
        # acc = compute_accuracy(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            loss, model, opt_state = make_step(model, *next(batches), opt_state)
            # print(f"loss: {loss}, acc: {acc}")
            wandb.log({"train/loss": loss})
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")

        train_acc = compute_accuracy(model, train_images, train_labels)
        test_acc = compute_accuracy(model, train_images, train_labels)
        # train_acc = compute_loader_accuracy(model, train_loader)
        # test_acc = compute_loader_accuracy(model, test_loader)
        wandb.log({"train/accuracy": train_acc})
        wandb.log({"test/accuracy": test_acc})
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")
