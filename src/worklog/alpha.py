from functools import partial
from itertools import islice
from typing import Any

import jax
import jax.numpy
import optax

from .core.hmm import (
    InterleavedHiddenMarkovChain,
    interleaved_ergodic_hidden_markov_chain,
)


class WorkLogAlpha:
    hmm: InterleavedHiddenMarkovChain
    variables: Any

    def __init__(self, key, num_actions):
        self.hmm = interleaved_ergodic_hidden_markov_chain(4, 4, num_actions)
        self.variables = self.hmm.init(key, key, jax.numpy.array([0]))
        self.state = None

    def forward(self, y):
        return self.hmm.apply(self.variables, y, method=self.hmm.forward)

    def fit(self, dataset, batch_size=16):
        optimizer = optax.adam(learning_rate=1e-3)
        self.state = self.state or optimizer.init(self.variables)

        model = self.hmm

        @partial(jax.vmap, in_axes=(None, 0))
        def forward(variables, y):
            return model.apply(variables, y, method=model.forward)

        @jax.value_and_grad
        def objective(variables, y):
            return -forward(variables, y).mean()

        @jax.jit
        def step(state, variables, o):
            loss, grads = objective(variables, o)
            updates, state = optimizer.update(grads, state)
            variables = optax.apply_updates(variables, updates)
            return state, variables, loss

        def batch(generator, batch_size):
            while True:
                b = [*islice(generator, 0, batch_size)]
                if len(b) == 0:
                    break
                yield jax.tree_map(lambda *x: jax.numpy.stack(x), *b)

        dataset = batch(dataset, batch_size)

        # training loop
        for o, _ in dataset:
            self.state, self.variables, loss = step(self.state, self.variables, o)
            yield loss
