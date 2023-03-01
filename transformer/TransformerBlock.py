import torch
import torch.nn as nn


# SelfAttentionBlock
class SelfAttention(nn.Module):
    def __init__(self, embedding_size, heads):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads

        assert (self.heads * self.head_dim == embedding_size), 'divided error'

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embedding_size, bias=False)

    def forward(self, query, key, values, mask):
        N = query.shape[0]  # batch_size
        query_len, key_len, value_len = query.shape[1], key.shape[1], values.shape[1]

        # split embedding into self.heads heads
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # compute attention = softmax( Qk' / (d)**(1/2) )
        # dim=3=key_len
        # attention shape:[N, heads, queries_len, keys_len]
        attention = torch.softmax(energy / (self.embedding_size ** (1 / 2)), dim=3)

        # compute attention*V
        # flatten last 2 dim
        out = torch.einsum("nhqk, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


# Build Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):  # times of hidden size of FFN
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.FFN = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        out = self.dropout(self.norm1(attention + query))

        FFN_out = self.FFN(out)
        output = self.dropout(self.norm2(FFN_out + out))

        return output
