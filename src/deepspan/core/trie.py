from typing import Protocol

class Trie[K](Protocol):
    def subtrie(self, key: K) -> "Trie[K]":
        ...

    def probability(self, key: K) -> float:
        ...

