import torch
import torch.nn as nn
import json


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        self.vocab_size = len(vocab) + 1
        hidden_size = 512
        self.word_embed = nn.Embedding(self.vocab_size, hidden_size)
        self.lstm_embed = nn.Embedding(self.vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

    def forward(self, seqs1, seqs2):
        dot = self._embed(seqs1) * self._embed(seqs2)
        outputs = torch.sigmoid(self.output_layer(dot)).squeeze(1)
        return outputs

    def _embed(self, seqs):
        word_embed = self._word_embed(seqs)
        lstm_embed = self._lstm_embed(seqs)
        embed = torch.cat([word_embed, lstm_embed], 1)
        return embed

    def _word_embed(self, seqs):
        masks = seqs > 0
        outputs = self.word_embed(seqs)
        outputs = outputs * masks.unsqueeze(2).float()
        outputs = outputs.sum(1)
        return self._norm(outputs)

    def _lstm_embed(self, seqs):
        lengths = (seqs > 0).sum(1)
        lengths[lengths == 0] = 1
        embed = self.lstm_embed(seqs)
        outputs, _ = self.lstm(embed)
        outputs = [outputs[b, s - 1, :] for b, s in enumerate(lengths)]
        outputs = torch.cat(outputs).view(len(seqs), -1)
        return self._norm(outputs)

    def _norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        y = (x - mean) / (std + 1e-8)
        return y

