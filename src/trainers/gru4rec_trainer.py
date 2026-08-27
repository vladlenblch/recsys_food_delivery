import torch
import numpy as np
from tqdm import tqdm
from src.metrics.ranking import Metrics, compute_rank

class GRU4RecTrainer():
    def __init__(self, model, bpr_loss, optimizer, vocab_size, device, num_negs=20):
        self.model = model
        self.bpr_loss = bpr_loss
        self.optimizer = optimizer
        self.vocab_size = vocab_size
        self.device = device
        self.num_negs = num_negs

    def sample_negatives(self, pos_indices):
        batch_size = pos_indices.size(0)
        negs = torch.randint(1, self.vocab_size, (batch_size, self.num_negs), device=self.device)

        mask = (negs == pos_indices.unsqueeze(1))
        while mask.any():
            new_negs = torch.randint(1, self.vocab_size, (mask.sum().item(),), device=self.device)
            negs[mask] = new_negs
            mask = (negs == pos_indices.unsqueeze(1))
        return negs

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for padded_inputs, padded_targets, lengths, mask in tqdm(dataloader, desc="Training"):
            padded_inputs = padded_inputs.to(self.device)
            padded_targets = padded_targets.to(self.device)
            mask = mask.to(self.device)

            outputs = self.model(padded_inputs)  # (batch, seq_len, embed_dim)

            flat_mask = mask.reshape(-1)  # (batch * seq_len,)
            if not flat_mask.any():
                continue

            flat_outputs = outputs.reshape(-1, outputs.size(-1))[flat_mask]  # (total_valid, embed_dim)
            flat_targets = padded_targets.reshape(-1)[flat_mask]             # (total_valid,)

            neg_indices = self.sample_negatives(flat_targets)                # (total_valid, num_negs)

            pos_embs = self.model.get_item_embeddings(flat_targets)          # (total_valid, embed_dim)
            neg_embs = self.model.get_item_embeddings(neg_indices)           # (total_valid, num_negs, embed_dim)

            loss = self.bpr_loss(flat_outputs, pos_embs, neg_embs)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_steps += 1

        return total_loss / total_steps

    @torch.no_grad()
    def evaluate(self, split_data, target_key, k=5, batch_size=256):
        self.model.eval()
        recalls, ndcgs, mrrs = [], [], []

        all_embeddings = self.model.get_item_embeddings(
            torch.arange(self.vocab_size, device=self.device)
        )  # (vocab_size, embed_dim)

        items = list(split_data.items())  # (user_id, data)
        total_users = len(items)

        for start in range(0, total_users, batch_size):
            batch_items = items[start:start + batch_size]
            batch_seqs = []
            batch_true_indices = []
            batch_lengths = []

            for user_id, data in batch_items:
                seq = data["train"]
                if len(seq) == 0:
                    continue
                batch_seqs.append(seq)
                batch_true_indices.append(data[target_key])
                batch_lengths.append(len(seq))

            if not batch_seqs:
                continue

            max_len = max(batch_lengths)
            batch_size_current = len(batch_seqs)
            padded_seqs = torch.zeros(batch_size_current, max_len, dtype=torch.long, device=self.device)
            for i, seq in enumerate(batch_seqs):
                padded_seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

            outputs = self.model(padded_seqs)  # (batch, max_len, embed_dim)

            hidden_list = []
            for i, length in enumerate(batch_lengths):
                hidden_list.append(outputs[i, length - 1, :])  # (embed_dim,)
            hidden_batch = torch.stack(hidden_list, dim=0)     # (batch, embed_dim)

            scores = torch.mm(all_embeddings, hidden_batch.T)  # (vocab_size, batch)

            for i, true_idx in enumerate(batch_true_indices):
                score_vector = scores[:, i].cpu().numpy()
                rank = compute_rank(score_vector, true_idx)
                recalls.append(Metrics.recall_at_k(rank, k))
                ndcgs.append(Metrics.ndcg_at_k(rank, k))
                mrrs.append(Metrics.mrr(rank))

        return {
            f"Recall@{k}": np.mean(recalls),
            f"NDCG@{k}": np.mean(ndcgs),
            "MRR": np.mean(mrrs)
        }
