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
        self.word_embed = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.lstm_embed1 = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.lstm1 = nn.LSTM(args.input_encoding_size, args.rnn_size, num_layers=1, batch_first=True)
        self.lstm_embed2 = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.lstm2 = nn.LSTM(args.input_encoding_size, args.rnn_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(args.rnn_size * 2 + args.input_encoding_size, 1)

    def forward(self, seqs1, seqs2):
        word_dot = self._word_embed(seqs1) * self._word_embed(seqs2)
        lstm_dot = self._lstm_embed(seqs1) * self._lstm_embed(seqs2)
        lstm_output = self._lstm_embed(seqs2, is_first=False)
        outputs = torch.cat([word_dot, lstm_dot, lstm_output], 1)
        outputs = torch.sigmoid(self.output_layer(outputs)).squeeze(1)
        return outputs

    def _word_embed(self, seqs):
        masks = seqs > 0
        outputs = self.word_embed(seqs)
        outputs = outputs * masks.unsqueeze(2).float()
        outputs = outputs.sum(1)
        return outputs

    def _lstm_embed(self, seqs, is_first=True):
        lengths = (seqs > 0).sum(1)
        lengths[lengths == 0] = 1
        if is_first:
            embed = self.lstm_embed1(seqs)
            outputs, _ = self.lstm1(embed)
        else:
            embed = self.lstm_embed2(seqs)
            outputs, _ = self.lstm2(embed)
        outputs = [outputs[b, s - 1, :] for b, s in enumerate(lengths)]
        outputs = torch.cat(outputs).view(len(seqs), -1)
        return outputs
