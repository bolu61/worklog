from typing import Protocol


class Trie[K](Protocol):
    def __getitem__(self, key: K) -> "Trie[K]": ...

    def __contains__(self, key: K) -> bool: ...
