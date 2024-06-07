from collections.abc import Callable, Generator, Iterable
from itertools import islice

from .core.trie import Trie


def match[
    K
](trie: Trie[K], seq: list[K]) -> Generator[int, None, None]:
    i = -1
    j = 0
    while i < j and j < len(seq):
        if seq[j] not in trie:
            return
        trie = trie[seq[j]]
        yield j
        i = j

        for k in trie.keys:
            try:
                j = seq.index(k, i + 1)
                break
            except ValueError:
                continue



def extract[T](idx: list[int], seq: list[T]) -> list[T]:
    out = []
    for i in reversed(idx):
        out.append(seq.pop(i))
    return [*reversed(out)]


def separate[
    T, K
](trie: Trie[K], seq: Iterable[T], maxlen: int, key: Callable[[T], K] = lambda x: x) -> Generator[
    list[T], None, None
]:
    seq_list: list[T] = [*seq]
    while len(seq_list) > 0:
        keys = [key(s) for s in seq_list[:maxlen]]
        extracted = extract([*match(trie, keys)], seq_list)
        if len(extracted) <= 0:
            seq_list.pop(0)
            continue
        yield extracted


if __name__ == "__main__":
    from .core.prefixspan import prefixspan

    db = [[1, 2, 3], [2, 2, 3], [2, 3, 1], [3, 2, 1], [2, 1, 1]]

    trie = prefixspan(db, minsup=3)

    for seq in separate(trie, [2, 2, 3, 1], 3, lambda x: x):
        print(seq)
