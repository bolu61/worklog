# %%
import csv
from collections import defaultdict
from typing import Sequence, cast

import jax
import jax.numpy as jnp
from experiments.datasets.dataset import dataset


def masked_csv_dataset(key, path, size, length):
    action_map = defaultdict(lambda: len(action_map))
    with open(path, "r") as f:
        data = jnp.array([action_map[a] for _, _, a in csv.reader(f)], dtype=jnp.uint32)

    data = data[: (len(data) // length) * length]
    data = data.reshape(-1, length)

    data = jax.random.choice(key, data, (size,), replace=False, axis=0)

    return dataset(data=cast(Sequence[jax.Array], data))


# %%
ds = masked_csv_dataset(
    jax.random.key(0),
    "/mnt/c/Users/anana5/OneDrive/Desktop/apache_james_load.csv",
    10_000,
    10,
)
