from collections.abc import Callable
from functools import wraps
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
    def logits_init(prob_initializer):
        @wraps(prob_initializer)
        def wrapper(key, *args, **kwargs):
            probs = prob_initializer(key, *args, **kwargs)
            probs = jnp.clip(probs, a_min=1e-8, a_max=1)
            probs = jnp.log(probs)
            return probs

        return wrapper

    @staticmethod
    @logits_init
    def uniform_choice_unitializer(key, interleaving):
        return jax.random.uniform(key, shape=(interleaving,))

    @staticmethod
    @logits_init
    def uniform_transition_initializer(key, interleaving, states):
        return jax.random.uniform(
            key, shape=(interleaving, states, states), minval=0, maxval=1
        )

    @staticmethod
    @logits_init
    def uniform_emission_initializer(key, interleaving, states, alphabet):
        return lax.map(
            lambda key: jax.random.permutation(
                key, jax.random.permutation(key, jnp.eye(states, alphabet)), axis=1
            ),
            jax.random.split(key, interleaving),
        )

    @staticmethod
    @logits_init
    def stationary_initializer(key, transition):
        transition = jnp.exp(nn.log_softmax(transition))
        eigvalues, eigvectors = jnp.linalg.eig(jnp.transpose(transition, (1, 0)))
        solution = jnp.isclose(eigvalues, 1)
        # it will be scaled accordingly using softmax
        return jnp.real(eigvectors)[:, jnp.argmax(solution)]

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
        self.transition = self.param(
            "transition",
            self.transition_initializer,
            self.interleaving,
            self.states,
        )
        self.emission = self.param(
            "emission",
            self.emission_initializer,
            self.interleaving,
            self.states,
            self.alphabet,
        )
        self.choice = self.param("choice", self.choice_initializer, self.interleaving)

        self.prior = self.param(
            "prior",
            jax.vmap(self.stationary_initializer, in_axes=(None, 0)),
            self.transition,
        )

        self.cprior = self.param("cprior", self.stationary_initializer, self.choice)

    def __call__(self, key, s):
        ckey, tkey, ekey = jax.random.split(key, 3)
        # choose a chain
        c = jnp.exp(nn.log_softmax(self.choice))
        i = jax.random.choice(ckey, self.interleaving, p=c)

        # compute new state of chosen chain
        t = jnp.exp(nn.log_softmax(self.transition[i, s[i]]))
        s = s.at[i].set(jax.random.choice(tkey, self.states, p=t))

        e = jnp.exp(nn.log_softmax(self.emission[i, s[i]]))
        o = jax.random.choice(ekey, self.alphabet, p=e)
        return (s, i), o

    def sample(self, key):
        """sample from the stationary distribution"""
        p = jnp.exp(nn.log_softmax(self.prior, axis=-1))
        return jax.vmap(lambda key, p: jax.random.choice(key, self.states, p=p))(
            jax.random.split(key, len(p)), p
        )

    def forward(self, ys):
        ### dim names: (choice, state, state) or (choice, state, alphabet)
        choice = jax.nn.log_softmax(self.choice)
        transition = jax.nn.log_softmax(self.transition, axis=-1)
        emission = jax.nn.log_softmax(self.emission, axis=-1)
        prior = jax.nn.log_softmax(self.prior, axis=-1)
        index = cprod(
            *[jnp.arange(self.states)] * self.interleaving,
            jnp.arange(self.interleaving),
        )

        def a(x, x_new, alpha):
            s, s_new, i = x[:-1], x_new[:-1], x_new[-1]
            return (
                alpha
                + choice[i]
                + transition[i, x[i], x_new[i]]
                + jnp.sum(jnp.log(s == s_new))
            )

        def b(x, y):
            s, c = x[:-1], x[-1]
            return emission[c, s[c], y]

        alpha = jnp.sum(cprod(*prior, choice), -1)

        for y in ys:
            alpha = jax.lax.map(
                lambda x_new: b(x_new, y)
                + jax.nn.logsumexp(
                    jax.vmap(a, in_axes=(0, None, 0))(index, x_new, alpha)
                ),
                index,
            )

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
