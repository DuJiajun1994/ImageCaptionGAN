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
        self.rnn_size = args.rnn_size
        self.num_layers = args.num_layers
        self.embedding = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.lstm = nn.LSTM(args.input_encoding_size, args.rnn_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(args.rnn_size * 2, 1)
        self.word_output_layer = nn.Linear(args.rnn_size, self.vocab_size)

    def forward(self, seqs1, seqs2):
        embed1 = self._embed_seqs(seqs1)
        embed2 = self._embed_seqs(seqs2)
        embed = torch.cat([torch.abs(embed1 - embed2), embed1 * embed2], 1)
        outputs = torch.sigmoid(self.output_layer(embed)).squeeze(1)
        return outputs

    def _embed_seqs(self, seqs):
        lengths = (seqs > 0).sum(1)
        lengths[lengths == 0] = 1
        embed = self.embedding(seqs)
        outputs, _ = self.lstm(embed)
        outputs = [outputs[b, s - 1, :] for b, s in enumerate(lengths)]
        outputs = torch.cat(outputs).view(len(seqs), -1)
        return outputs
