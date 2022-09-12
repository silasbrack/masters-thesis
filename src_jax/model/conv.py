import equinox as eqx
import jax
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp


class ConvNet(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    fc1: nn.Linear
    fc2: nn.Linear
    # do1: nn.Dropout
    # do2: nn.Dropout
    # key: jax.random.PRNGKey

    def __init__(self, key: jax.random.PRNGKey):
        # self.key = key
        self.conv1 = nn.Conv2d(1, 32, 3, 1, key=key)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, key=key)
        self.fc1 = nn.Linear(1600, 128, key=key)
        self.fc2 = nn.Linear(128, 10, key=key)
        # self.do1 = nn.Dropout(0.25)
        # self.do2 = nn.Dropout(0.5)

    def __call__(self, x: jnp.array) -> jnp.array:
        x = self.conv1(x)
        x = jnn.relu(x)
        x = nn.MaxPool2d(2, stride=2)(x)
        # x = self.do1(x, key=self.key)
        x = jnn.relu(x)
        x = self.conv2(x)
        x = jnn.relu(x)
        x = nn.MaxPool2d(2, stride=2)(x)
        # x = self.do1(x, key=self.key)
        x = jnp.ravel(x)
        x = self.fc1(x)
        x = jnn.relu(x)
        # x = self.do2(x, key=self.key)
        x = self.fc2(x)
        x = jnn.log_softmax(x)
        return x
        # return reduce(
        #     lambda x, f: f(x),
        #     [
        #         self.conv1,
        #         jnn.relu,
        #         nn.MaxPool2d(2, stride=2),
        #         partial(self.do1, key=self.key),
        #         jnn.relu,
        #         self.conv2,
        #         jnn.relu,
        #         nn.MaxPool2d(2, stride=2),
        #         partial(self.do1, key=self.key),
        #         jnp.ravel,
        #         self.fc1,
        #         jnn.relu,
        #         partial(self.do2, key=self.key),
        #         self.fc2,
        #         jnn.log_softmax,
        #     ],
        #     x,
        # )
