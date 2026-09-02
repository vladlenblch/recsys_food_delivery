import torch
import numpy as np
from tqdm import tqdm
from src.metrics.ranking import Metrics

class SASRecTrainer():
    def __init__(self, model, bpr_loss, optimizer, vocab_size, device, num_negs=10):
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

            padding_mask = ~mask
            outputs = self.model(padded_inputs, padding_mask=padding_mask)

            flat_mask = mask.reshape(-1)
            if not flat_mask.any():
                continue

            flat_outputs = outputs.reshape(-1, outputs.size(-1))[flat_mask]
            flat_targets = padded_targets.reshape(-1)[flat_mask]

            neg_indices = self.sample_negatives(flat_targets)

            pos_embs = self.model.get_item_embeddings(flat_targets)
            neg_embs = self.model.get_item_embeddings(neg_indices)

            loss = self.bpr_loss(flat_outputs, pos_embs, neg_embs)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_steps += 1

        return total_loss / total_steps

    @torch.no_grad()
    def evaluate(self, split_data, target_key, k=5, batch_size=32):
        self.model.eval()

        all_embeddings = self.model.get_item_embeddings(
            torch.arange(1, self.vocab_size, device=self.device)
        )  # (vocab_size - 1, embed_dim)

        items = list(split_data.items())
        total_users = len(items)

        all_ranks = []

        for start in tqdm(range(0, total_users, batch_size), desc=f"Evaluating {target_key}"):
            batch_items = items[start:start + batch_size]
            batch_seqs = []
            batch_true_indices = []
            batch_lengths = []

            for user_id, data in batch_items:
                seq = data["train"]
                if len(seq) == 0:
                    continue

                seq = seq[-self.model.max_len:]

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

            padding_mask = (padded_seqs == 0)
            outputs = self.model(padded_seqs, padding_mask=padding_mask)  # (batch, max_len, embed_dim)

            lengths_tensor = torch.tensor(batch_lengths, device=self.device) - 1
            batch_idx = torch.arange(batch_size_current, device=self.device)
            hidden_batch = outputs[batch_idx, lengths_tensor]  # (batch, embed_dim)

            scores = torch.mm(all_embeddings, hidden_batch.T)  # (vocab_size - 1, batch)

            true_indices_tensor = torch.tensor(batch_true_indices, device=self.device)
            true_positions = true_indices_tensor - 1

            true_scores = scores[true_positions, batch_idx]  # (batch,)

            ranks = (scores > true_scores.unsqueeze(0)).sum(dim=0) + 1  # (batch,)

            all_ranks.append(ranks.cpu().numpy())

        all_ranks = np.concatenate(all_ranks)

        recalls = (all_ranks <= k).astype(np.float32)
        ndcgs = np.where(all_ranks <= k, 1.0 / np.log2(all_ranks + 1), 0.0)
        mrrs = 1.0 / all_ranks

        return {
            f"Recall@{k}": float(recalls.mean()),
            f"NDCG@{k}": float(ndcgs.mean()),
            "MRR": float(mrrs.mean())
        }
