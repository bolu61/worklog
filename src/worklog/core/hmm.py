from collections.abc import Callable
from functools import partial, wraps
from typing import Optional

import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp


@jax.jit
def cprod(*a):
    return jnp.stack(jnp.meshgrid(*a, indexing="ij"), -1).reshape(-1, len(a))


@jax.jit
def log1mexp(x):
    return jnp.where(x < jnp.log(0.5), jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


class InterleavedHiddenMarkovChain(nn.Module):
    @staticmethod
    def uniform_choice_unitializer(key, interleaving):
        return jax.random.uniform(key, shape=(interleaving,))

    @staticmethod
    def uniform_transition_initializer(key, interleaving, states):
        return jax.random.uniform(
            key, shape=(interleaving, states, states), minval=0, maxval=1
        )

    @staticmethod
    def uniform_emission_initializer(key, interleaving, states, alphabet):
        return lax.map(
            lambda key: jax.random.permutation(
                key, jax.random.permutation(key, jnp.eye(states, alphabet)), axis=1
            ),
            jax.random.split(key, interleaving),
        )

    interleaving: int
    states: int
    alphabet: int
    choice_initializer: Callable[
        [jax.Array, int], jax.Array
    ] = uniform_choice_unitializer
    transition_initializer: Callable[
        [jax.Array, int, int], jax.Array
    ] = uniform_transition_initializer
    emission_initializer: Callable[
        [jax.Array, int, int, int], jax.Array
    ] = uniform_emission_initializer

    def __hash__(self):
        return hash(id(self))

    def setup(self):
        def logits_init(prob_initializer):
            @wraps(prob_initializer)
            def wrapper(key, *args, **kwargs):
                probs = prob_initializer(key, *args, **kwargs)
                probs = jnp.clip(probs, a_min=1e-8, a_max=1)
                probs = jnp.log(probs)
                return probs

            return wrapper

        self.transition = self.param(
            "transition",
            logits_init(self.transition_initializer),
            self.interleaving,
            self.states,
        )
        self.emission = self.param(
            "emission",
            logits_init(self.emission_initializer),
            self.interleaving,
            self.states,
            self.alphabet,
        )
        self.choice = self.param(
            "choice", logits_init(self.choice_initializer), self.interleaving
        )

        # calculate stationary distribution
        @logits_init
        def stationary_initializer(key, t):
            t = jax.nn.softmax(t)
            eigvalues, eigvectors = jnp.linalg.eig(jnp.transpose(t, (0, 2, 1)))
            solution = jnp.isclose(eigvalues, 1)
            p = jnp.real(eigvectors)[
                jnp.arange(self.interleaving), :, jnp.argmax(solution, axis=1)
            ]
            p = p / jnp.sum(p)
            return p

        self.prior = self.param("prior", stationary_initializer, self.transition)

    def __call__(self, key, s):
        ckey, tkey, ekey = jax.random.split(key, 3)
        # choose a chain
        c = jax.nn.softmax(self.choice)
        i = jax.random.choice(ckey, self.interleaving, p=c)

        # compute new state of chosen chain
        t = jax.nn.softmax(self.transition[i, s[i]])
        s = s.at[i].set(jax.random.choice(tkey, self.states, p=t))

        e = jax.nn.softmax(self.emission[i, s[i]])
        o = jax.random.choice(ekey, self.alphabet, p=e)
        return (s, i), o

    def sample(self, key):
        """sample from the stationary distribution"""
        p = jax.nn.softmax(self.prior)
        _, s = lax.scan(
            lambda i, key: (i + 1, jax.random.choice(key, self.states, p=p[i])),
            0,
            jax.random.split(key, self.interleaving),
        )
        return s

    @nn.jit
    def sforward(self, ys, cs):
        ### dim names: (choice, state, state) or (choice, state, alphabet)
        c = jax.nn.log_softmax(self.choice)
        t = jax.nn.log_softmax(self.transition)
        e = jax.nn.log_softmax(self.emission)
        a = jax.nn.log_softmax(self.prior)

        for y, i in zip(ys, cs):
            a = a.at[i].set(e[i, :, y] + c[i] + nn.logsumexp(t[i] + a[i, :, jnp.newaxis], axis=0))

        return nn.logsumexp(a.sum(axis=0))

    @nn.jit
    def forward(self, ys):
        ### dim names: (choice, state, state) or (choice, state, alphabet)
        c = jax.nn.log_softmax(self.choice)
        t = jax.nn.log_softmax(self.transition)
        e = jax.nn.log_softmax(self.emission)
        p = jax.nn.log_softmax(self.prior)
        i = cprod(
            *[jnp.arange(self.states)] * self.interleaving,
            jnp.arange(self.interleaving),
        )

        @partial(jax.vmap, in_axes=(None, 0))
        @partial(jax.vmap, in_axes=(0, None))
        def a(x, x_new):
            i = x_new[-1]
            s = x[:-1]
            s_new = x_new[:-1]
            p = t[i, x[i], x_new[i]]
            a = c[i]
            return a + jnp.sum(jnp.log(s == s_new).at[i].set(p))

        @partial(jax.vmap, in_axes=(0, None))
        def b(x, y):
            return e[x[-1], x[x[-1]], y]

        alpha = jnp.sum(cprod(*p, c), -1)

        for y in ys:
            alpha = b(i, y) + jax.nn.logsumexp(alpha + a(i, i), axis=-1)

        return jax.nn.logsumexp(alpha)


def interleaved_ergodic_hidden_markov_chain(
    interleaving: int, states: int, alphabet: Optional[int], shape=1
):
    """Random Ergodic Hidden Markov Chain
    transition weights are sampled from a beta distribution.
    """
    if alphabet is None:
        alphabet = states

    shape = jnp.clip(shape, 0.1, 0.9)

    def transition_initializer(key, interleaving, states):
        key, subkey = jax.random.split(key)
        t = jax.random.beta(
            subkey, a=1 - shape, b=shape, shape=(interleaving, states, states)
        )
        t = jnp.clip(t, a_min=1e-8, a_max=1)
        t = t / jnp.sum(t, axis=2, keepdims=True)
        return t

    return InterleavedHiddenMarkovChain(
        interleaving, states, alphabet, transition_initializer=transition_initializer
    )


def interleaved_cyclic_markov_chain(
    interleaving, states: int, alphabet: Optional[int] = None
):
    if alphabet is None:
        alphabet = states

    def transition_initializer(key, interleaving, states):
        t = jnp.eye(states)
        t = jnp.roll(t, 1, axis=-1)
        t = jnp.repeat(t[jnp.newaxis, :, :], interleaving, axis=0)
        return t

    return InterleavedHiddenMarkovChain(
        interleaving, states, alphabet, transition_initializer=transition_initializer
    )
