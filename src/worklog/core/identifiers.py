from collections.abc import MutableSequence, Sequence

from prefixspan import prefixspan
from functools import partial


def propagate(trie: prefixspan, sequence, ids):
    """Propagate IDs according to the input prefixspan trie and sequence.
    IDs are positive non-zero integers; an ID of 0 indicates an unknown class.
    """
    for i in range(len(sequence)):
        if ids[i] == 0:
            continue

        t = trie

        for j in range(i, len(sequence)):
            if t.empty():
                break

            if sequence[j] not in t or (
                ids[j] != ids[i] and ids[j] != 0
            ):
                continue

            t = t[sequence[j]]
            ids[j] = ids[i]
