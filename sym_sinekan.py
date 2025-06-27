from flax import linen as nn
from jax._src.lax.control_flow.loops import X
import jax.numpy as jnp
import jax

from typing import Tuple


class SineKANLayer(nn.Module):
    input_dim: int
    output_dim: int
    grid_size: int = 8
    is_first: bool = False
    add_bias: bool = True
    norm_freq: bool = True

    def setup(self):
        # Precompute grid_phase and input_phase as they depend only on grid_size and input_dim
        self.grid_phase = jnp.linspace(0, jnp.pi, self.grid_size).reshape(1, self.grid_size)
        self.input_phase = jnp.linspace(0, jnp.pi, self.input_dim).reshape(self.input_dim, 1)

        # Combine phases to get (input_dim, grid_size)
        self.phase = self.input_phase + self.grid_phase  # Shape: (input_dim, grid_size)

        # Add random perturbation to break symmetries of phase
        key = jax.random.PRNGKey(42)  # Random seed is explicit in JAX
        grid_phase_perturbation = jax.random.uniform(key, shape=(1, self.grid_size)) / (self.grid_size) * 0.1
        input_phase_perturbation = jax.random.uniform(key, shape=(self.input_dim, 1)) / (self.input_dim) * 0.1

        self.phase = self.phase + grid_phase_perturbation + input_phase_perturbation

        # Save the phase for debugging if needed
        self.sow("phase", "buffer", self.phase)

        # Compute freq_init based on norm_freq and is_first
        if self.norm_freq:
            freq_init = (jnp.arange(1, self.grid_size + 1) / (self.grid_size + 1) ** (1 - self.is_first)).reshape(1, self.grid_size)
        else:
            freq_init = (jnp.arange(1, self.grid_size + 1) / (self.grid_size + 1)).reshape(1, self.grid_size)
        
        # Broadcast freq_init to (input_dim, grid_size) and ensure correct dtype
        freq_init = jnp.broadcast_to(freq_init, (self.input_dim, self.grid_size)).astype(jnp.float32)  # Assuming float32; adjust if needed

        # Initialize 'freq' parameter once
        self.freq = self.param('freq', nn.initializers.constant(freq_init), (self.input_dim, self.grid_size))

        # Initialize the Dense layer once
        self.dense = nn.Dense(
            features=self.output_dim,
            use_bias=self.add_bias,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.zeros if self.add_bias else None,
            param_dtype=jnp.float32,  # Adjust dtype if necessary
            dtype=jnp.float32         # Adjust dtype if necessary
        )

    def __call__(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.shape[0]

        # Compute sine activations:
        # x: (batch_size, input_dim)
        # freq: (input_dim, grid_size)
        # phase: (input_dim, grid_size)
        # s: (batch_size, input_dim, grid_size)
        s = jnp.sin(x[:, :, None] * self.freq[None, :, :] + self.phase[None, :, :])

        # Flatten the (input_dim, grid_size) into a single dimension
        # s_flat: (batch_size, input_dim * grid_size)
        s_flat = s.reshape(batch_size, self.input_dim * self.grid_size)

        # Apply the Dense layer
        y = self.dense(s_flat)  # (batch_size, output_dim)

        return y

class SineKAN(nn.Module):
    layers_hidden: Tuple[int, ...]
    grid_size: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Extract the input dimension dynamically from x
        input_dim = x.shape[-1]
        # Construct the full hidden layer dimensions including the input dimension
        layers_hidden = (input_dim,) + self.layers_hidden

        # Iterate over consecutive pairs of layers
        for i, (in_dim, out_dim) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            x = SineKANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                grid_size=self.grid_size,
                is_first=(i == 0)
            )(x)

        if x.shape[-1] == 1:
            return x.squeeze(-1)
        else:
            return x
        
class SymmetricSineKAN1D(nn.Module):
    layers_hidden: Tuple[int, ...]
    grid_size: int = 8
    pbc: bool = False

    def setup(self):
        self.amplitude_sinekan = SineKAN(layers_hidden=self.layers_hidden, grid_size=self.grid_size)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_ref = jnp.flip(x, axis=-1)

        # Compute symmetric transformations for amplitude
        amp_val = self.amplitude_sinekan(x)
        amp_val_ref = self.amplitude_sinekan(x_ref)

        amplitude = amp_val + amp_val_ref

        return amplitude