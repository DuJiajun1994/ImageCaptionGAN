import torch
import torch.nn as nn
import json
from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        hidden = 768
        n_layers = 4
        attn_heads = 12
        dropout = 0.1
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        vocab_size = len(vocab) + 1
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden * 2, 1)

    def forward(self, seqs1, seqs2):
        embed1 = self._embed_seqs(seqs1)
        embed2 = self._embed_seqs(seqs2)
        embed = torch.cat([torch.abs(embed1 - embed2), embed1 * embed2], 1)
        outputs = torch.sigmoid(self.output_layer(embed)).squeeze(1)
        return outputs

    def _embed_seqs(self, seqs):
        mask = (seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)
        x = self.embedding(seqs)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x[:, 0]
