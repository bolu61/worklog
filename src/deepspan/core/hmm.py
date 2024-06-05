from typing import Any, Optional

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
    interleaving: int
    states: int
    alphabet: int
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
            (self.interleaving, self.states, self.states),
            jnp.float32,
        )
        self.emission = self.param(
            "emission",
            self.emission_initializer,
            (self.interleaving, self.states, self.alphabet),
            jnp.float32,
        )
        self.choice = self.param(
            "choice", self.choice_initializer, (self.interleaving,), jnp.float32
        )

        self.prior = self.param(
            "prior",
            self.prior_initializer,
            (self.interleaving, self.states),
            jnp.float32,
        )

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
