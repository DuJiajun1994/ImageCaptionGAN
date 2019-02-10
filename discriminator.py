import torch
import torch.nn as nn
import json
from lstm_embedding import LSTMEmbedding


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        vocab_size = len(vocab) + 1
        hidden_size = 512
        self.word_embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm_embed1 = LSTMEmbedding(vocab_size=vocab_size, rnn_size=hidden_size)
        self.lstm_embed2 = LSTMEmbedding(vocab_size=vocab_size, rnn_size=hidden_size)
        self.lstm_embed3 = LSTMEmbedding(vocab_size=vocab_size, rnn_size=hidden_size)
        self.fc_embed = nn.Linear(args.fc_feat_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size * 4, 1)

    def forward(self, fc_feats, seqs1, seqs2):
        txt2txt = self._embed(seqs1) * self._embed(seqs2)
        im2txt = self._norm(self.fc_embed(fc_feats)) * self._norm(self.lstm_embed2(seqs2))
        txt = self.lstm_embed3(seqs2)
        outputs = torch.sigmoid(self.output_layer(torch.cat([txt2txt, im2txt, txt], 1))).squeeze(1)
        return outputs

    def _embed(self, seqs):
        word_embed = self._norm(self._word_embed(seqs))
        lstm_embed = self._norm(self.lstm_embed1(seqs))
        embed = torch.cat([word_embed, lstm_embed], 1)
        return embed

    def _word_embed(self, seqs):
        masks = seqs > 0
        outputs = self.word_embed(seqs)
        outputs = outputs * masks.unsqueeze(2).float()
        outputs = outputs.sum(1)
        return outputs

    def _norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        y = (x - mean) / (std + 1e-8)
        return y
