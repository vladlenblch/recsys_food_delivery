import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        
        if hidden_dim != embed_dim:
            self.projection = nn.Linear(in_features=hidden_dim, out_features=embed_dim, bias=False)
        else:
            self.projection = None

    def forward(self, input_seq):
        embs = self.embedding(input_seq)
        output, _ = self.gru(embs)
        if self.projection is not None:
            output = self.projection(output)

        return output

    def get_item_embeddings(self, item_indices):
        return self.embedding(item_indices)
