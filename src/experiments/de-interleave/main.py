from worklog.core.hmm import interleaved_ergodic_hidden_markov_chain
from prefixspan import prefixspan
from functools import partial
from itertools import islice
from array import array

from ..datasets.dataset import dataset

import jax


def masked_process_dataset(key, size, interleaving, states, shape, length):
    """Synthetic dataset of interleaved sequences
    Uses an interleaved_arcsin_markov_chain to generate the sequences.
    """
    mc = interleaved_ergodic_hidden_markov_chain(interleaving, states, states, shape)

    key, init_subkey = jax.random.split(key)
    variables = mc.init(init_subkey, jax.random.key(0), jax.numpy.array([0]))

    @jax.jit
    def sequence(key):
        sample_subkey, sequence_subkey = jax.random.split(key, 2)
        state = mc.apply(variables, sample_subkey, method=mc.sample)

        def wrapper(s, key):
            (s, c), y = mc.apply(variables, key, s)
            return s, (y, c)

        _, (y, _) = jax.lax.scan(
            wrapper, state, jax.random.split(sequence_subkey, length)
        )
        return y

    keys = jax.random.split(key, size)

    return dataset([sequence(k) for k in keys])


def propagate(trie: prefixspan, sequence):
    mask = [False for _ in sequence]

    def rec(t, i):
        while i < len(sequence) and sequence[i] not in t:
            i += 1

        if not i < len(sequence):
            return 0
            
        if (a := rec(t[sequence[i]], i + 1)) > (b := rec(t, i + 1)):
            print("a")
            mask[i] = True
            return a + 1
        else:
            print("b")
            return b

    mask[0] = True
    rec(trie[sequence[0]], 1)

    return mask


key = jax.random.key(0xc0ffee)

key, key_1 = jax.random.split(key)

ds = masked_process_dataset(
    key=key,
    size=1000,
    interleaving=3,
    states=16,
    shape=1,
    length=16
)
ds = ds.map(lambda s: array("L", s))

train_ds, test_ds = ds.split(999)
trie = prefixspan(train_ds, 8)

print(test_ds[0])
print(propagate(trie, test_ds[0]))

