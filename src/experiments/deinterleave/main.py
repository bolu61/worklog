from collections.abc import Sequence
from pathlib import Path
from random import sample

import pandas as pd

from deepspan.core.prefixspan import prefixspan
from deepspan.separate import separate

df = pd.read_csv(Path(__file__).parent / "datasets" / "Hadoop_2k.log_structured.csv")

train_seq = df.EventId
eval_seq = df.to_records()[:1000]


def make_database[T](seq: Sequence[T], size: int, stride: int) -> list[list[T]]:
    def windows():
        for i in range(0, len(seq), stride):
            yield [*seq[i : i + size]]

    return list(windows())


db = sample(make_database(train_seq, size=16, stride=16), 600 // 16)

trie = prefixspan(db, minsup=16)

for separated_seq in separate(trie, eval_seq, key=lambda x: x.EventId):
    print([(s.LineId, s.EventId) for s in separated_seq])
