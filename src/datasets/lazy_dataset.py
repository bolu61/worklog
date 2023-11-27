from dataclasses import dataclass
from typing import Callable, Generator, Generic, Sequence, TypeVar

from jax.typing import ArrayLike

T = TypeVar("T")


@dataclass
class LazyDataset(Generic[T]):
    keys: Sequence[ArrayLike]
    func: Callable[[ArrayLike], T]

    def __init__(self, keys, func):
        self.keys = keys
        self.func = func

    def __iter__(self) -> Generator[T, None, None]:
        for key in self.keys:
            yield self.func(key)

    def __getitem__(self, index) -> T:
        return self.func(self.keys[index])

    def split(self, count) -> tuple["LazyDataset[T]", "LazyDataset[T]"]:
        return LazyDataset(self.keys[:count], self.func), LazyDataset(
            self.keys[count:], self.func
        )
