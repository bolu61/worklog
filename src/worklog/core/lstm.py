from typing import Any
import flax.linen as nn


class SequenceMixtureRNNLayer(nn.Module):
    cell: Any
    n_heads: int
    n_dims: int

    def setup(self):
        self.layer = nn.scan(
            target=self.cell,
            variable_broadcast="params",
            split_rngs={"params": False},
        )

    def __call__(self, x):
        return x
