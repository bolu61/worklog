import pickle
from functools import partial
from itertools import islice
from typing import Any, Optional

import jax
import numpy as np
import optax
from scipy.stats import mannwhitneyu

from .core.hmm import (
    InterleavedHiddenMarkovChain,
    interleaved_ergodic_hidden_markov_chain,
)


class WorkLogAlpha:
    hmm: InterleavedHiddenMarkovChain
    variables: Optional[Any]
    states: Optional[Any]

    def __init__(self, action_count):
        self.hmm = interleaved_ergodic_hidden_markov_chain(4, 4, action_count)
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

        @partial(jax.vmap, in_axes=(None, 0))
        def forward(variables, y):
            return model.apply(variables, y, method=model.forward)

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
        for o, _ in dataset:
            self.state, self.variables, loss = step(self.state, self.variables, o)
            yield loss

    def evaluate(self, key, dataset):
        mc = self.hmm
        variables = self.variables or self.hmm.init(key, key, jax.numpy.array([0]))
        @jax.jit
        def sequence(key, length):
            sample_subkey, sequence_subkey = jax.random.split(key, 2)
            state = mc.apply(variables, sample_subkey, method=mc.sample)

            @jax.jit
            def wrapper(s, key):
                (s, c), y = mc.apply(variables, key, s)
                return s, (y, c)

            _, (y, c) = jax.lax.scan(
                wrapper, state, jax.random.split(sequence_subkey, length)
            )
            return y, c

        totals = np.zeros((self.hmm.alphabet, len(dataset)), dtype=np.uint32)
        totals_pred = np.zeros((self.hmm.alphabet, len(dataset)), dtype=np.uint32)
        for i, (os, _) in enumerate(dataset):
            for o in os:
                totals[o, i] += 1
            for o in sequence(key, len(os)):
                totals_pred[o, i] += 1

        return np.mean(np.abs(totals - totals_pred), axis=1), mannwhitneyu(totals, totals_pred, axis=1)

    def dump(self, file):
        pickle.dump({"state": self.state, "variables": self.variables}, file)

    def load(self, file):
        data = pickle.load(file)
        self.variables = data["variables"]
        self.state = data["state"]
