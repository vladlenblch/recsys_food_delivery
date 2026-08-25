import numpy as np
from collections import Counter

class PopularityBaseline():
    def __init__(self):
        self.popularity_scores = None
        self.vocab_size = 0

    def fit(self, sequences, vocab_size):
        self.vocab_size = vocab_size

        counter = Counter()
        for seq in sequences.values():
            counter.update(seq)

        self.popularity_scores = np.zeros(self.vocab_size, dtype=np.float32)
        for idx, count in counter.items():
            if idx < self.vocab_size:
                self.popularity_scores[idx] = count

    def score(self, session=None):
        return self.popularity_scores.copy()
