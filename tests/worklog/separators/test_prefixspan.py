from array import array
from prefixspan import prefixspan
from worklog.core.identifiers import propagate


def sliding_window(sequence, width, stride):
    for i in range(len(sequence) // stride):
        start = i * stride
        yield sequence[start : start + width]


def test_simple():
    o = array('L', [2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0, 0, 1])
    i = array('L', [1, 1, 0, 1, 2, 2, 0, 1, 1, 0, 1, 1, 0, 2, 0, 0])

    db = list(sliding_window(o, 3, 1))
    ps = prefixspan(db, len(db) // 3)
    propagate(ps, o, i)
    assert i == array('L', [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2])
