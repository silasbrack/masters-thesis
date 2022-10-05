import logging
import os
import time
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import optax
from dotenv import load_dotenv
from jax import numpy as jnp
from jax import random
from jax.tree_util import Partial, tree_map

import wandb
from src_jax.data.dataloader import data_stream
from src_jax.data.mnist import mnist
from src_jax.model import ConvNet, FcNet
from src_jax.utils import compute_accuracy


def cross_entropy_loss(targets, preds, axis=-1):
    return -jnp.mean(jnp.sum(preds * targets, axis=axis))


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    load_dotenv()
    wandb.init(project=os.getenv("WANDB_PROJECT"))

    data_dir = os.getenv("DATA_DIR")

    step_size = 0.001
    num_epochs = 3
    batch_size = 512
    pretrained_path = Path("pretrained")
    pretrained_path.mkdir(exist_ok=True, parents=True)
    seed = 42
    model_cls = ConvNet  # FcNet
    posterior_samples = 10

    model_key, train_key, data_key = random.split(random.PRNGKey(seed), 3)

    model = model_cls(model_key)

    num_params = sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(model))
    logging.info(f"Number of parameters: {num_params}")

    train_images, train_labels, test_images, test_labels = mnist(data_dir)
    train_images = jnp.array(train_images)
    train_labels = jnp.array(train_labels)
    test_images = jnp.array(test_images)
    test_labels = jnp.array(test_labels)
    num_train = len(train_images)

    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    batches = data_stream(train_images, train_labels, batch_size, key=data_key)
    optim = optax.adam(step_size)
    opt_state = optim.init(model)

    def compute_loss(params, static, inputs, targets, **kwargs):
        model_fn = Partial(eqx.combine(params, static), **kwargs)
        preds = jax.vmap(model_fn, in_axes=0)(inputs)
        loss = cross_entropy_loss(targets, preds)
        return loss

    @partial(jax.jit, static_argnums=1)
    def make_step(params, static, x, y, opt_state, **kwargs):
        loss, grads = jax.value_and_grad(compute_loss)(params, static, x, y, **kwargs)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    params, static = eqx.partition(model, eqx.is_array)

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(num_batches):
            x, y = next(batches)
            loss, params, opt_state = make_step(params, static, x, y, opt_state, key=train_key)
            wandb.log({"train/loss": loss})
            train_key, _ = random.split(train_key)

        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch} in {epoch_time:0.2f} sec")

        # Right now, calculating the accuracy while training takes like 5x longer than the training itself.
        # start_time = time.time()
        # model_fn = Partial(model, key=train_key)
        # train_acc = compute_accuracy(model_fn, train_images, train_labels)
        # test_acc = compute_accuracy(model_fn, test_images, test_labels)
        # accuracy_time = time.time() - start_time
        # logging.info(f"Accuracy in {accuracy_time:0.2f} sec")
        # wandb.log({"train/accuracy": train_acc})
        # wandb.log({"test/accuracy": test_acc})
        # logging.info(f"Training set accuracy {train_acc}")
        # logging.info(f"Test set accuracy {test_acc}")

    model = eqx.combine(params, static)

    eqx.tree_serialise_leaves(pretrained_path / "pretrained.eqx", model)
    artifact = wandb.Artifact("convnet", type="model")
    artifact.add_file(pretrained_path / "pretrained.eqx")
    wandb.log_artifact(artifact)

    model = eqx.tree_deserialise_leaves(pretrained_path / "pretrained.eqx", model)
    model_fn = Partial(model, inference=True, key=None)
    train_acc = compute_accuracy(model_fn, train_images, train_labels)
    test_acc = compute_accuracy(model_fn, test_images, test_labels)
    logging.info(f"Training set accuracy {train_acc}")
    logging.info(f"Test set accuracy {test_acc}")

    # Only calculate Hessian for the linear layers
    filter_spec = tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.fc1.weight, tree.fc1.bias, tree.fc2.weight, tree.fc2.bias),
        filter_spec,
        replace=(True, True, True, True),
    )
    params, static = eqx.partition(model, filter_spec)
    hessian = jax.hessian(compute_loss)(params, static, x, y, inference=True, key=None)
    print(hessian)
    print("Hessian of second fully-connected layer weights wrt first fully-connected layer bias:", hessian.fc2.weight.fc1.bias.shape)
    print("This is equivalent to {fc2_out}x{fc2_in}x{fc1_out}")
    # The issue with this is that the Hessian is calculated as a PyTree, which means it's not that easy to sample from.



    # diagonal_filter = tree_map(lambda _: False, hessian)
    # diagonal_filter = eqx.tree_at(
    #     lambda tree: (tree.fc1.weight.fc1.weight, tree.fc1.bias.fc1.bias, tree.fc2.weight.fc2.weight, tree.fc2.bias.fc2.bias),
    #     diagonal_filter,
    #     replace=(True, True, True, True),
    # )
    # diag_hessian, _ = eqx.partition(hessian, diagonal_filter)
    # def extract_diag(x):
    #     if x.ndim == 2:
    #         return jnp.einsum("ii->i", x)
    #     elif x.ndim == 4:
    #         return jnp.einsum("ijij->ij", x)
    #     else:
    #         raise ValueError(f"Unexpected shape {x.shape}")
    # diag_hessian = tree_map(extract_diag, diag_hessian)



    # def get_posterior_scale(param):
    #     scale = 1.0
    #     prior_prec = 1.0
    #     posterior_precision = param * scale + prior_prec
    #     posterior_scale = 1.0 / (jnp.sqrt(posterior_precision) + 1e-6)
    #     return posterior_scale

    # # posterior_scale = tree_map(get_posterior_scale, diag_hessian)

    # samples = jax.random.normal(train_key, (posterior_samples, *posterior_scale.shape))
    # samples = samples * posterior_scale 



    # Calculating the GGN Hessian is kinda funky
    # 
    # @partial(jax.jit, static_argnums=1)
    # def model_fn(params, static, x):
    #     model = eqx.combine(params, static)
    #     return jax.vmap(model)(x)

    # batch_size = x.shape[0]
    # output_size = 10
    # def jac_prod(jac):
    #     jac = jac.reshape((batch_size, output_size, -1))
    #     prod = jnp.einsum("ijk,imn->ikn", jac, jac)
    #     return jnp.einsum("ijj->ij", prod)

    # nn_jacobian = jax.jacfwd(model_fn)(params, static, x)
    # ggn_hessian = jax.tree_util.tree_map(jac_prod, nn_jacobian)
