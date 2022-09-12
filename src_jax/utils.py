import equinox as eqx
import jax
from jax import numpy as jnp


def compute_loader_accuracy(model, loader):
    matches = 0
    total = 0
    for inputs, targets in loader:
        preds = jax.vmap(model, in_axes=0)(inputs)
        preds = jnp.argmax(preds, axis=1)
        targets = jnp.argmax(targets, axis=1)
        match = jnp.sum(preds == targets)
        matches += match
        total += len(inputs)
    return matches / total


@eqx.filter_jit
def compute_accuracy(model, inputs, targets):
    target_class = jnp.argmax(targets, axis=1)
    preds = jax.vmap(model, in_axes=0)(inputs)
    predicted_class = jnp.argmax(preds, axis=1)
    return jnp.mean(predicted_class == target_class)
