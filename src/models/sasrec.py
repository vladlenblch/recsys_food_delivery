import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, num_layers=2, nhead=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.item_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=256,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, input_seq, padding_mask=None):
        batch_size, seq_len = input_seq.size()

        item_embs = self.item_embeddings(input_seq) # (batch, seq_len, embed_dim)

        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0) # (1, seq_len)

        pos_embs = self.positional_embeddings(positions) # (1, seq_len, embed_dim)

        embs = item_embs + pos_embs

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=input_seq.device
            ), 
            diagonal=1
        ).bool()

        if padding_mask is None:
            padding_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=input_seq.device
            )

        output = self.transformer(
            embs,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        return output

    def get_item_embeddings(self, item_indices):
        return self.item_embeddings(item_indices)
