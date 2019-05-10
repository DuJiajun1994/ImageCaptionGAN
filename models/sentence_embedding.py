import torch
import torch.nn as nn
from models.lstm_embedding import LSTMEmbedding
from models.word_embedding import WordEmbedding


class SentenceEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SentenceEmbedding, self).__init__()
        self.word = WordEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)
        self.lstm = LSTMEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)

    def forward(self, seqs):
        word = self._norm(self.word(seqs))
        lstm = self._norm(self.lstm(seqs))
        embed = torch.cat([word, lstm], 1)
        return embed

    def _norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        y = (x - mean) / (std + 1e-8)
        return y
