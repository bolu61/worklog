from collections.abc import Generator, Iterable, Sequence
from functools import partial
from typing import Any, Optional
from itertools import islice

import flax.linen as nn
import jax
import jax.numpy as jnp


@jax.jit
def cprod(*a):
    return jnp.stack(jnp.meshgrid(*a, indexing="ij"), -1).reshape(-1, len(a))


@jax.jit
def log1mexp(x):
    return jnp.where(x < jnp.log(0.5), jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


class InterleavedHiddenMarkovChain(nn.Module):
    num_chains: int
    num_states: int
    num_symbols: int
    choice_initializer: Any = nn.initializers.uniform()
    prior_initializer: Any = nn.initializers.glorot_uniform()
    transition_initializer: Any = nn.initializers.glorot_uniform()
    emission_initializer: Any = nn.initializers.glorot_uniform()

    def __hash__(self):
        return hash(id(self))

    def setup(self):
        self.transition = self.param(
            "transition",
            self.transition_initializer,
            (self.num_chains, self.num_states, self.num_states),
            jnp.float32,
        )
        self.emission = self.param(
            "emission",
            self.emission_initializer,
            (self.num_chains, self.num_states, self.num_symbols),
            jnp.float32,
        )
        self.choice = self.param(
            "choice", self.choice_initializer, (self.num_chains,), jnp.float32
        )

        self.prior = self.param(
            "prior",
            self.prior_initializer,
            (self.num_chains, self.num_states),
            jnp.float32,
        )

    def __call__(self, key, s):
        ckey, tkey, ekey = jax.random.split(key, 3)
        # choose a chain
        c = jnp.exp(nn.log_softmax(self.choice))
        i = jax.random.choice(ckey, self.num_chains, p=c)

        # compute new state of chosen chain
        t = jnp.exp(nn.log_softmax(self.transition[i, s[i]]))
        s = s.at[i].set(jax.random.choice(tkey, self.num_states, p=t))

        e = jnp.exp(nn.log_softmax(self.emission[i, s[i]]))
        o = jax.random.choice(ekey, self.num_symbols, p=e)
        return (s, i), o

    def sample(self, key):
        """sample from the stationary distribution"""
        p = jnp.exp(nn.log_softmax(self.prior, axis=-1))
        return jax.vmap(lambda key, p: jax.random.choice(key, self.num_states, p=p))(
            jax.random.split(key, len(p)), p
        )


def interleaved_ergodic_hidden_markov_chain(
    interleaving: int, states: int, alphabet: Optional[int], shape=1
):
    """Random Ergodic Hidden Markov Chain
    transition weights are sampled from a beta distribution.
    """
    if alphabet is None:
        alphabet = states

    shape = jnp.clip(shape, 0.1, 0.9)

    def transition_initializer(key, shape_, dtype):
        key, subkey = jax.random.split(key)
        t = jnp.log(
            jax.random.beta(
                subkey,
                a=1 - shape,
                b=shape,
                shape=shape_,
                dtype=dtype,
            )
        )
        return t

    def emission_initializer(key, shape, dtype):
        interleaving, states, alphabet = shape
        e = jax.vmap(
            lambda key: jax.random.permutation(
                key, jnp.eye(states, alphabet, dtype=dtype)
            )
        )(jax.random.split(key, interleaving))
        return jnp.clip(jnp.log(e), -1e8)

    return InterleavedHiddenMarkovChain(
        interleaving,
        states,
        alphabet,
        transition_initializer=transition_initializer,
        emission_initializer=emission_initializer,
    )


def interleaved_cyclic_markov_chain(
    interleaving, states: int, alphabet: Optional[int] = None
):
    if alphabet is None:
        alphabet = states

    def transition_initializer(key, shape, dtype):
        interleaving, states, states = shape
        t = jnp.eye(states, dtype=dtype)
        t = jnp.roll(t, 1, axis=-1)
        t = jnp.repeat(t[jnp.newaxis, :, :], interleaving, axis=0)
        return jnp.clip(jnp.log(t), -1e8)

    def emission_initializer(key, shape, dtype):
        interleaving, states, alphabet = shape
        e = jax.vmap(
            lambda key: jax.random.permutation(
                key, jnp.eye(states, alphabet, dtype=dtype)
            )
        )(jax.random.split(key, interleaving))
        return jnp.clip(jnp.log(e), -1e8)

    return InterleavedHiddenMarkovChain(
        interleaving,
        states,
        alphabet,
        transition_initializer=transition_initializer,
        emission_initializer=emission_initializer,
    )


def interleaved_markov_chain(num_chains, num_states):
    def apply_log(f):
        def wrapped(*args):
            return jnp.log(f(*args))
        return wrapped

    @apply_log
    def transition_initializer(_, shape, dtype):
        num_chains, num_states, num_states = shape
        m = jnp.eye(num_states, dtype=dtype)
        m = jnp.roll(m, 1, axis=-1)
        return jnp.tile(m, (num_chains, 1, 1))

    @apply_log
    def emission_initializer(_, shape, dtype):
        num_chains, num_states, num_symbols = shape
        def make_matrix(i):
            eye = jnp.eye(num_states, num_symbols, dtype=dtype)
            return jnp.roll(eye, i * num_states, axis=-1)
        return jax.vmap(make_matrix)(jnp.arange(num_chains))

    return InterleavedHiddenMarkovChain(
        num_chains=num_chains,
        num_states=num_states,
        num_symbols=num_chains * num_states,
        transition_initializer=transition_initializer,
        emission_initializer=emission_initializer
    )
