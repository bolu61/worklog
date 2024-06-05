from collections import defaultdict
from collections.abc import Sequence

from .trie import trie

type Database[T] = Sequence[Sequence[T]]


def project[T](db: Database[T], s: T) -> Database[T]:
    p: Database[T] = []
    for seq in db:
        for t in (iter_seq := iter(seq)):
            if t == s:
                p.append([*iter_seq])
    return p


def prefixspan[T](db: Database[T], minsup: int) -> trie[T, int]:
    t = trie[T, int](len(db))

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
    print(t.find([2, 3]).get())
