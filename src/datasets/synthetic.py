import jax
from worklog.core.hmm import interleaved_ergodic_hidden_markov_chain

from .lazy_dataset import LazyDataset


def interleaved_ergodic_process_dataset(
    key, size, interleaving, states, alphabet, shape, length
):
    """Synthetic dataset of interleaved sequences
    Uses an interleaved_arcsin_markov_chain to generate the sequences.
    """
    mc = interleaved_ergodic_hidden_markov_chain(interleaving, states, alphabet, shape)

    key, init_subkey = jax.random.split(key)
    variables = mc.init(init_subkey, jax.random.key(0), jax.numpy.array([0]))

    @jax.jit
    def sequence(key):
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

    return LazyDataset(jax.random.split(key, size), sequence)
