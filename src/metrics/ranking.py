import numpy as np


def compute_rank(scores, true_index):
    if len(scores) == 0:
        return 0

    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    elif not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    sorted_indices = np.argsort(scores)[::-1]

    positions = np.where(sorted_indices == true_index)[0]
    if len(positions) == 0:
        return 0
    return positions[0] + 1


class Metrics():
    @staticmethod
    def recall_at_k(true_rank, k):
        if true_rank <= k:
            return 1.0
        return 0.0

    @staticmethod
    def ndcg_at_k(true_rank, k):
        if true_rank <= k:
            return 1.0 * 1 / np.log2(true_rank + 1)
        return 0.0

    @staticmethod
    def mrr(true_rank):
        if true_rank > 0:
            return 1.0 * 1 / true_rank
        return 0.0
