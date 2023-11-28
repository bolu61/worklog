from dataclasses import dataclass
from typing import Any, Callable, Generator, Generic, Sequence, TypeVar, Union

from jax.typing import ArrayLike

S = TypeVar("S")
T = TypeVar("T")


@dataclass
class LazyDataset(Generic[T]):
    keys: Sequence[ArrayLike]
    func: Callable[[ArrayLike], T]

    def __iter__(self) -> Generator[T, None, None]:
        for key in self.keys:
            yield self.func(key)

    def __getitem__(self, index) -> T:
        return self.func(self.keys[index])

    def __len__(self):
        return len(self.keys)

    def split(self, count) -> tuple["LazyDataset[T]", "LazyDataset[T]"]:
        return LazyDataset(self.keys[:count], self.func), LazyDataset(
            self.keys[count:], self.func
        )


@dataclass
class MappedDataset(Generic[S, T]):
    dataset: Union[LazyDataset[S], "MappedDataset[Any, S]"]
    func: Callable[[S], T]

    def __iter__(self):
        for data in self.dataset:
            yield self.func(data)

    def __getitem__(self, index) -> T:
        return self.func(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def split(self, count) -> tuple["MappedDataset[S, T]", "MappedDataset[S, T]"]:
        a, b = self.dataset.split(count)
        return MappedDataset(a, self.func), MappedDataset(b, self.func)
