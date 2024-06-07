from pathlib import Path
from random import sample

import pandas as pd

from pandas import Series

from deepspan.core.prefixspan import prefixspan
from deepspan.separate import separate

df = pd.read_csv(Path(__file__).parent / "datasets" / "OpenSSH_2k.log_structured.csv")

df.index = pd.to_datetime(df.Date + " " + df.Time)

df = df.sort_index()

def make_database(seq: Series, window_size, min_length) -> list[list[object]]:
    def windows():
        for w in seq.rolling(window_size, min_periods=min_length):
            yield w.values.tolist()

    return list(windows())


db = sample(make_database(df.EventId, window_size="8s", min_length=2), 2000)

trie = prefixspan(db, minsup=int(len(db) * 0.2))

for separated_seq in separate(trie, df.to_records(), maxlen=16, key=lambda x: x.EventId):
    print([(s.LineId, s.EventId) for s in separated_seq])
