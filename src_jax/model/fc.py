import equinox as eqx
import jax
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp

class FcNet(eqx.Module):
    fc1: nn.Linear
    fc2: nn.Linear
    # key: jax.random.PRNGKey

    def __init__(self, key: jax.random.PRNGKey):
        # self.key = key
        self.fc1 = nn.Linear(784, 11, key=key)
        self.fc2 = nn.Linear(11, 10, key=key)

    def __call__(self, x: jnp.array) -> jnp.array:
        x = jnp.ravel(x)
        x = self.fc1(x)
        x = jnn.relu(x)
        x = self.fc2(x)
        x = jnn.log_softmax(x)
        return x

# class FcNet(eqx.Module):
#     fc1: nn.Linear
#     fc2: nn.Linear
#     fc3: nn.Linear
#     key: jax.random.PRNGKey

#     def __init__(self, key: jax.random.PRNGKey):
#         self.key = key
#         self.fc1 = nn.Linear(784, 1024, key=key)
#         self.fc2 = nn.Linear(1024, 1024, key=key)
#         self.fc3 = nn.Linear(1024, 10, key=key)

#     def __call__(self, x: jnp.array) -> jnp.array:
#         x = jnp.ravel(x)
#         x = self.fc1(x)
#         x = jnn.relu(x)
#         x = self.fc2(x)
#         x = jnn.relu(x)
#         x = self.fc3(x)
#         x = jnn.log_softmax(x)
#         return x
