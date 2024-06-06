from collections.abc import Callable, Generator, Iterable

from .core.trie import Trie


def extract[
    T, K
](trie: Trie[K], seq: list[T], key: Callable[[T], K]) -> Generator[T, None, None]:
    i = 0
    while i < len(seq):
        if key(seq[i]) not in trie:
            i += 1
            continue
        yield (s := seq.pop(i))
        trie = trie[key(s)]


def separate[
    T, K
](trie: Trie[K], seq: Iterable[T], key: Callable[[T], K] = lambda x: x) -> Generator[
    list[T], None, None
]:
    m: list[T] = [*seq]
    while len(m) > 0:
        extracted = [*extract(trie, m, key)]
        if len(extracted) <= 0:
            break
        yield extracted


if __name__ == "__main__":
    from .core.prefixspan import prefixspan

    db = [[1, 2, 3], [2, 2, 3], [2, 3, 1], [3, 2, 1], [2, 1, 1]]

    trie = prefixspan(db, minsup=3)

    for seq in separate(trie, [2, 2, 3, 1], lambda x: x):
        print(seq)
