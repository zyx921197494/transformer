import torch
import torch.nn as nn
from TransformerBlock import *


# Build Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(src_vocab_size, embed_size)
        self.position_encoding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask):
        N, seq_len = X.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # out is actually inputs after embedding and positional encoding
        out = self.dropout(self.word_embed(X) + self.position_encoding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
