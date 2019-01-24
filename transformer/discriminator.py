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
        self.cos = nn.CosineSimilarity()

    def forward(self, seqs1, seqs2):
        embed1 = self._embed_seqs(seqs1)
        embed2 = self._embed_seqs(seqs2)
        outputs = (self.cos(embed1, embed2) + 1) / 2
        return outputs

    def _embed_seqs(self, seqs):
        masks = seqs > 0
        att_masks = masks.unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)
        outputs = self.embedding(seqs)
        for transformer in self.transformer_blocks:
            outputs = transformer.forward(outputs, att_masks)
        outputs = outputs * masks.unsqueeze(2).float()
        length = masks.sum(1)
        length[length == 0] = 1
        outputs = outputs.sum(1) / length.unsqueeze(1).float()
        return outputs
