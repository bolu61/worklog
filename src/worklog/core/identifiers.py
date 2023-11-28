from array import array

from prefixspan import prefixspan


def propagate(trie: prefixspan, sequence, ids):
    """Propagate IDs according to the input prefixspan trie and sequence.
    IDs are positive non-zero integers; an ID of -1 indicates an unknown class.
    """
    ids = array('i', ids)
    for i in range(len(sequence)):
        if ids[i] == -1:
            ids[i] = 0

        t = trie

        for j in range(i, len(sequence)):
            if t.empty():
                break

            if sequence[j] in t and (ids[i] == ids[j] or ids[j] == -1):
                t = t[sequence[j]]
                ids[j] = ids[i]

    return ids
