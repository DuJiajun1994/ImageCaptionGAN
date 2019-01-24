import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        self.vocab_size = len(vocab) + 1
        self.embedding = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.cos = nn.CosineSimilarity()

    def forward(self, seqs1, seqs2):
        embed1 = self._embed_seqs(seqs1)
        embed2 = self._embed_seqs(seqs2)
        outputs = (self.cos(embed1, embed2) + 1) / 2
        return outputs

    def _embed_seqs(self, seqs):
        masks = seqs > 0
        outputs = self.embedding(seqs)
        outputs = outputs * masks.unsqueeze(2).float()
        outputs = outputs.sum(1)
        return outputs
