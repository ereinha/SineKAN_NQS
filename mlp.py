from flax import linen as nn
from typing import Tuple
import jax.numpy as jnp


class MLP(nn.Module):
    layers_hidden: Tuple[int, ...]
    grid_size: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Iterate over consecutive pairs of layers
        for i, out_dim in enumerate(self.layers_hidden):
            x = nn.Dense(
                features=out_dim,
            )(x)
            if i < len(self.layers_hidden) - 1:
                x = nn.relu(x)
        return x.squeeze(-1)

class SymmetricMLP(nn.Module):
    layers_hidden: Tuple[int, ...]
    grid_size: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        layers = []
        x_flip = jnp.flip(x, axis=-1)
        # Iterate over consecutive pairs of layers
        for i, out_dim in enumerate(self.layers_hidden):
            layers.append(nn.Dense(
                features=out_dim,
              ))
        for i, layer in enumerate(layers):
            x = layer(x)
            if i < len(self.layers_hidden) - 1:
                x = nn.relu(x)
        for i, layer in enumerate(layers):
            x_flip = layer(x_flip)
            if i < len(self.layers_hidden) - 1:
                x_flip = nn.relu(x_flip)
        return x.squeeze(-1) + x_flip.squeeze(-1)