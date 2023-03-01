import torch
import torch.nn as nn
from TransformerBlock import *


# Build Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = (self.dropout(self.norm(attention + x)))
        out = self.transformer_block(query, key, value, src_mask)
        return out


# Build Decoder
class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(target_vocab_size, embed_size)
        self.position_encoding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, dropout, forward_expansion, device) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, enc_out, src_mask, target_mask):
        N, seq_len = X.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        X = self.dropout(self.word_embed(X) + self.position_encoding(positions))

        for layer in self.layers:
            X = layer(X, enc_out, enc_out, src_mask, target_mask)
        out = self.fc_out(X)

        return out
