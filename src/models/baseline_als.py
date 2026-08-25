import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

class ALSBaseline():
    def __init__(self, factors=64, regularization=0.01, iterations=20, alpha=40):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self.model = None
        self.user2idx = None
        self.vocab_size = 0

    def fit(self, sequences, vocab_size):
        self.vocab_size = vocab_size
        self.user2idx = {uid: i for i, uid in enumerate(sequences.keys())}
        num_users = len(self.user2idx)

        rows = []
        cols = []
        data = []

        for user_id, seq in sequences.items():
            u_idx = self.user2idx[user_id]

            for item in set(seq):
                if item < self.vocab_size:
                    rows.append(u_idx)
                    cols.append(item)
                    data.append(1.0)

        matrix = csr_matrix((data, (rows, cols)), shape=(num_users, vocab_size), dtype=np.float32)

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=42
        )
        self.model.fit(matrix)

    def score(self, user_idx):
        user_vec = self.model.user_factors[user_idx]
        item_vecs = self.model.item_factors
        scores = np.dot(item_vecs, user_vec)

        return scores.astype(np.float32)
