import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, seqs):
        masks = seqs > 0
        outputs = self.embedding(seqs)
        outputs = outputs * masks.unsqueeze(2).float()
        outputs = outputs.sum(1)
        return outputs
