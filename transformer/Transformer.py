import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx, embed_size=256, num_layers=6,
                 forward_expansion=4, heads=8, dropout=0, device='cpu', max_len=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout,
                               max_len)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout,
                               max_len)
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(N, 1, target_len, target_len)
        return target_mask

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)
        return out


if __name__ == '__main__':
    device = 'cpu'
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    target_pad_idx = 0
    src_vocab_size = 10
    target_vocab_size = 10
    model = Transformer(src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx).to(device)
    out = model(x, target[:, :-1])
    print(out.shape)
