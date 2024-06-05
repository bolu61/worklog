from collections.abc import Iterable, Iterator


class trie[K, T]:
    _children: dict[K, "trie[K, T]"]
    _data: T

    def __init__(self, data: T):
        self._children = dict()
        self._data = data

    def set(self, data: T):
        self._data = data

    def get(self) -> T:
        return self._data

    def find(self, str: Iterable[K]) -> "trie[K, T]":
        def _find(t: trie[K, T], it: Iterator[K]):
            while True:
                try:
                    k = next(it)
                except StopIteration:
                    return t
                t = t._children[k]

        return _find(self, iter(str))

    def insert(self, key, t: "trie[K, T]"):
        self._children[key] = t

    def __str__(self) -> str:
        return (
            "("
            + ",".join((str(k) + str(t) for k, t in self._children.items()))
            + "):"
            + str(self._data)
        )
