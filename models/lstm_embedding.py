import torch
import torch.nn as nn


class LSTMEmbedding(nn.Module):
    def __init__(self, vocab_size, rnn_size):
        super(LSTMEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, rnn_size)
        self.lstm = nn.LSTM(rnn_size, rnn_size, num_layers=1, batch_first=True)

    def forward(self, seqs):
        lengths = (seqs > 0).sum(1)
        lengths[lengths == 0] = 1
        embed = self.embedding(seqs)
        outputs, _ = self.lstm(embed)
        outputs = [outputs[b, s - 1, :] for b, s in enumerate(lengths)]
        outputs = torch.cat(outputs).view(len(seqs), -1)
        return outputs
