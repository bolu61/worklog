from collections import defaultdict
from collections.abc import Iterable, Iterator

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

    def find(self, str: Iterable[K]) -> "DictTrie[K]":
        def _find(t: DictTrie[K], it: Iterator[K]):
            while True:
                try:
                    k = next(it)
                except StopIteration:
                    return t
                t = t._children[k]

        return _find(self, iter(str))

    def insert(self, key, t: "DictTrie[K]"):
        self._children[key] = t

    def __str__(self) -> str:
        return (
            "("
            + ",".join((str(k) + str(t) for k, t in self._children.items()))
            + "):"
            + str(self._count)
        )

    def subtrie(self, key: K) -> "DictTrie[K]":
        return self._children[key]

    def probability(self, key: K) -> float:
        raise NotImplementedError()



def project[T](db: Database[T], s: T) -> Database[T]:
    p: Database[T] = []
    for seq in db:
        for t in (iter_seq := iter(seq)):
            if t == s:
                p.append([*iter_seq])
    return p


def prefixspan[T](db: Database[T], minsup: int) -> DictTrie[T]:
    t = DictTrie[T](len(db))

    count = defaultdict[T, int](lambda: 0)
    for seq in db:
        for s in set(seq):
            count[s] += 1

    for s, c in count.items():
        if c < minsup:
            continue
        proj: Database[T] = project(db, s)
        t.insert(s, prefixspan(proj, minsup))

    return t


if __name__ == "__main__":
    db = [[1, 2, 3], [2, 3, 1], [2, 2, 3], [2, 2, 2]]

    t = prefixspan(db, 3)

    print(t)
    print(t.find([2, 3])._count)
