from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator

import jax
import jax.numpy as jnp

from ..worklog.hmm import interleaved_ergodic_hidden_markov_chain, sequence


@dataclass
class GeneratorDataset:
    size: int
    generator: Iterator[Any]

    def __init__(self, size, generator):
        self.size = size
        self.generator = islice(generator, size)

    def __iter__(self):
        return self

    def __next__(self):
        self.size -= 1
        return next(self.generator)

    def __len__(self):
        return self.size

    def take(self, count):
        return [*islice(self, count)]


def dataset(key, size, interleaving, states, alphabet, shape, length):
    """Synthetic dataset of interleaved sequences
    Uses an interleaved_arcsin_markov_chain to generate the sequences.
    """
    mc = interleaved_ergodic_hidden_markov_chain(interleaving, states, alphabet, shape)

    key, init_subkey = jax.random.split(key)
    variables = mc.init(init_subkey, jax.random.key(0), jnp.array([0]))

    @jax.jit
    def sample(variables, key):
        return mc.apply(variables, key, method=mc.sample)

    @jax.jit
    def seq(variables, key, state):
        return mc.apply(variables, key, state, length, method=sequence)

    def generator(key):
        while True:
            key, sample_subkey, sequence_subkey = jax.random.split(key, 3)
            initial_states = sample(variables, sample_subkey)
            yield seq(variables, sequence_subkey, initial_states)

    return GeneratorDataset(size, generator(key))
