from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence

from .database import Database


class DictTrie[K]:
    _children: dict[K, "DictTrie[K]"]
    _count: int

    def __init__(self, count: int):
        self._children = dict()
        self._count = count

    @property
    def count(self):
        return self._count

    def find(self, seq: Sequence[K]) -> "DictTrie[K]":
        def _find(t: DictTrie[K], it: Iterator[K]):
            while True:
                try:
                    k = next(it)
                except StopIteration:
                    return t
                t = t._children[k]

        return _find(self, iter(seq))

    def insert(self, key, t: "DictTrie[K]"):
        self._children[key] = t

    def __str__(self) -> str:
        return (
            "("
            + ",".join((str(k) + str(t) for k, t in self._children.items()))
            + "):"
            + str(self._count)
        )

    def __getitem__(self, key: K) -> "DictTrie[K]":
        return self._children[key]

    def __contains__(self, key: K) -> bool:
        return key in self._children

    def prob(self, key: K) -> float:
        return self._children[key]._count / self._count

    @property
    def keys(self) -> list[K]:
        return sorted(self._children.keys(), reverse=True, key=lambda k: self._children[k]._count)


type Index = list[tuple[int, int]]


def project[
    T
](db: Database[T], index: Index, s: T) -> Generator[tuple[int, int], None, None]:
    for i, j in index:
        try:
            yield i, db[i].index(s, j) + 1
        except ValueError:
            continue


def prefixspan[T](db: Database[T], minsup: int) -> DictTrie[T]:
    def rec(index: Index):
        t = DictTrie[T](len(index))
        count = defaultdict[T, int](lambda: 0)
        for i, j in index:
            for s in set(db[i][j:]):
                count[s] += 1

        for s, c in count.items():
            if c < minsup:
                continue
            t.insert(s, rec([*project(db, index, s)]))

        return t

    return rec([(i, 0) for i in range(len(db))])
