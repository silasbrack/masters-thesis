import equinox as eqx
import jax
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp


class ConvNet(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    conv3: nn.Conv2d
    fc1: nn.Linear
    fc2: nn.Linear
    do1: nn.Dropout
    do2: nn.Dropout

    def __init__(self, key: jax.random.PRNGKey):
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1, key=key)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1, key=key)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0, key=key)
        self.fc1 = nn.Linear(3*3*32, 32, key=key)
        self.fc2 = nn.Linear(32, 10, key=key)
        self.do1 = nn.Dropout(0.15)
        self.do2 = nn.Dropout(0.25)

    def __call__(self, x: jnp.array, key: jax.random.PRNGKey, inference: bool = False) -> jnp.array:
        k1, k2, k3, k4 = jax.random.split(key, 4) if key is not None else (None, None, None, None)
        x = self.conv1(x)
        x = jnn.relu(x)
        x = self.conv2(x)
        x = jnn.relu(x)
        x = self.conv3(x)
        x = jnn.relu(x)
        x = jnp.ravel(x)
        x = self.fc1(x)
        x = jnn.relu(x)
        x = self.do2(x, key=k4, inference=inference)
        x = self.fc2(x)
        x = jnn.log_softmax(x)
        return x
