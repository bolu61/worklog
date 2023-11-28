import pickle
from functools import partial
from itertools import islice
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .core.hmm import (
    InterleavedHiddenMarkovChain,
)
from .core.identifiers import prefixspan, propagate


def fix(dataset, window=16, stride=12):
    def small_sequences():
        for o, _ in dataset:
            for i in range(0, len(o), stride):
                yield np.array(o[i : i + window], dtype=np.uint64)

    database = list(small_sequences())
    trie = prefixspan(database, len(database))

    for o, i in dataset:
        i = jnp.asarray(propagate(trie, o, i), dtype=jnp.uint32)
        o = jnp.asarray(o, dtype=jnp.uint32)
        yield o, i


class WorkLogBeta:
    hmm: InterleavedHiddenMarkovChain
    variables: Optional[Any]
    states: Optional[Any]

    def __init__(self, cluster_count, action_count):
        self.hmm = InterleavedHiddenMarkovChain(
            cluster_count, action_count, action_count
        )
        self.optimizer = optax.adamaxw(learning_rate=1)
        self.variables = None
        self.state = None

    def forward(self, y):
        return self.hmm.apply(self.variables, y, method=self.hmm.forward)

    def fit(self, key, dataset, batch_size=1):
        self.variables = self.variables or self.hmm.init(key, key, jax.numpy.array([0]))
        self.state = self.state or self.optimizer.init(self.variables)

        model = self.hmm
        optimizer = self.optimizer

        @partial(jax.vmap, in_axes=(None, 0, 0))
        def sforward(variables, o, i):
            return model.apply(variables, o, i, method=model.sforward)

        @jax.value_and_grad
        def objective(variables, o, i):
            return -sforward(variables, o, i).mean()

        @jax.jit
        def step(state, variables, o, i):
            loss, grads = objective(variables, o, i)
            updates, state = optimizer.update(grads, state, variables)
            variables = optax.apply_updates(variables, updates)
            return state, variables, loss

        def batch(generator, batch_size):
            while True:
                b = [*islice(generator, 0, batch_size)]
                if len(b) == 0:
                    break
                yield jax.tree_map(lambda *x: jax.numpy.stack(x), *b)

        dataset = batch(fix(dataset, 16, 12), batch_size)

        # training loop
        for o, i in dataset:
            self.state, self.variables, loss = step(self.state, self.variables, o, i)
            yield loss

    def dump(self, file):
        pickle.dump({"state": self.state, "variables": self.variables}, file)

    def load(self, file):
        data = pickle.load(file)
        self.variables = data["variables"]
        self.state = data["state"]
