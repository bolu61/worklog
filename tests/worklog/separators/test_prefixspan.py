from worklog.separators.prefixspan import separate

def test_separate():
    result = separate([0, 1, 2, 3, 4, 5], 3)
    assert list(result) == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5],
        [5]
    ]
