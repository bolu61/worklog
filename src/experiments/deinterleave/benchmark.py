from pathlib import Path
from random import sample
from typing import cast

import pandas as pd

from pandas import Series

from deepspan.core.prefixspan import prefixspan
from deepspan.separate import separate
from deepspan.core.hmm import interleaved_markov_chain

import jax

def make_interleaved_sequence(key, length, num_chains, num_states) -> tuple[jax.Array, jax.Array]:
    mc = interleaved_markov_chain(num_chains=num_chains, num_states=num_states)
    variables = mc.init(key, jax.random.key(0), jax.numpy.array([0]))


    sample_subkey, sequence_subkey = jax.random.split(key, 2)
    state = mc.apply(variables, sample_subkey, method=mc.sample)

    def gen_next(state, key):
        (state, choice), y = mc.apply(variables, key, state)
        return state, (y, choice)


    _, (seq, choices) = jax.lax.scan(
        gen_next, state, jax.random.split(sequence_subkey, length)
    )

    seq = cast(jax.Array, seq)

    return seq, choices

seq, _ = make_interleaved_sequence(key=jax.random.key(0), length=20480, num_chains=8, num_states=4)

train_seq = seq[:20480]
eval_seq = seq[:64]

db = train_seq.reshape(-1, 32).tolist()

trie = prefixspan(db, minsup=int(len(db) * 0.2))

print(trie, flush=True)

for proc in separate(trie=trie, seq=eval_seq.tolist(), maxlen=32):
    print(proc, flush=True)

