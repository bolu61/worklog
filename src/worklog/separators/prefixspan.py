def separate(sequence, max_len):
    def sliding_window(sequence, width, stride):
        for i in range(len(sequence) // stride):
            start = i * stride
            yield sequence[start : start + width]

    yield from list(sliding_window(sequence, max_len, 1))
