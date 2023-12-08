from functools import partial
from itertools import islice
from typing import Any, Optional, cast

import jax
import optax
from flax.core.scope import VariableDict
from jax import lax

from .core.hmm import (
    InterleavedHiddenMarkovChain,
)


class WorkLogAlpha:
    hmm: InterleavedHiddenMarkovChain
    variables: Optional[VariableDict]
    states: Optional[Any]

    def __init__(self, cluster_count, sequence_length, action_count, lr=1e-2):
        self.hmm = InterleavedHiddenMarkovChain(
            cluster_count, sequence_length, action_count
        )
        self.optimizer = optax.adamaxw(learning_rate=lr)
        self.cluster_count = cluster_count
        self.sequence_length = sequence_length
        self.action_count = action_count
        self.variables = None
        self.state = None

    @partial(jax.jit, static_argnums=(2,))
    def sequence(self, key, length):
        if self.variables is None:
            raise RuntimeError("model hasn't been fitted yet")
        v = self.variables

        def seq(s, k):
            (s, i), o = self.hmm.apply(v, k, s)
            return s, o

        sample_key, scan_key = jax.random.split(key)
        s = self.hmm.apply(self.variables, sample_key, method=self.hmm.sample)
        s, os = lax.scan(seq, s, jax.random.split(scan_key, length))
        return os

    @jax.jit
    def forward(self, y):
        if self.variables is None:
            raise RuntimeError("model hasn't been fitted yet")
        return cast(
            jax.Array, self.hmm.apply(self.variables, y, method=self.hmm.forward)
        )

    def fit(self, key, dataset, batch_size=1):
        self.variables = self.variables or self.hmm.init(key, key, jax.numpy.array([0]))
        self.state = self.state or self.optimizer.init(self.variables)

        model = self.hmm
        optimizer = self.optimizer

        @partial(jax.vmap, in_axes=(None, 0))
        def forward(variables, y) -> jax.Array:
            return cast(jax.Array, model.apply(variables, y, method=model.forward))

        @jax.value_and_grad
        def objective(variables, y):
            return -forward(variables, y).mean()

        @jax.jit
        def step(state, variables, o):
            loss, grads = objective(variables, o)
            updates, state = optimizer.update(grads, state, variables)
            variables = optax.apply_updates(variables, updates)
            return state, variables, loss

        def batch(generator, batch_size):
            while True:
                b = [*islice(generator, 0, batch_size)]
                if len(b) == 0:
                    break
                yield jax.tree_map(lambda *x: jax.numpy.stack(x), *b)

        dataset = batch(iter(dataset), batch_size)

        # training loop
        for os in dataset:
            self.state, self.variables, loss = step(self.state, self.variables, os)
            yield loss


def flatten(m):
    return (m.variables, m.state), (m.cluster_count, m.sequence_length, m.action_count)


def unflatten(aux, data):
    m = WorkLogAlpha(*aux)
    m.variables = data[0]
    m.state = data[1]
    return m


jax.tree_util.register_pytree_node(WorkLogAlpha, flatten, unflatten)
