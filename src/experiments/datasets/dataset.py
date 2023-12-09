from dataclasses import dataclass
from typing import Any, Callable, Generator, Generic, Sequence, TypeVar, Union

__all__ = ["dataset"]

S = TypeVar("S")
T = TypeVar("T")


def dataset(data: Sequence[T]) -> "Dataset[T]":
    return Dataset(data)


@dataclass
class Dataset(Generic[T]):
    data: Sequence[T]

    def __iter__(self) -> Generator[T, None, None]:
        yield from self.data

    def __getitem__(self, key) -> T:
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def split(self, count) -> tuple["Dataset[T]", "Dataset[T]"]:
        return Dataset(self.data[:count]), Dataset(self.data[count:])

    def map(self, f):
        return MappedDataset(self, f)


@dataclass
class MappedDataset(Generic[S, T]):
    dataset: Union[Dataset[S], "MappedDataset[Any, S]"]
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

    def map(self, f):
        return MappedDataset(self, f)
