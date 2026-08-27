import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, pos_embs, neg_embs):
        pos_scores = torch.sum(hidden * pos_embs, dim=1, keepdim=True)
        neg_scores = torch.bmm(neg_embs, hidden.unsqueeze(2)).squeeze(2)
        loss = -F.logsigmoid(pos_scores - neg_scores)

        return loss.mean()
