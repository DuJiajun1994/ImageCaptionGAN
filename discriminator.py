import torch
import torch.nn as nn
import json
from models.sentence_embedding import SentenceEmbedding


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        vocab_size = len(vocab) + 1
        self.hidden_size = args.rnn_size
        self.sentence_embed1 = SentenceEmbedding(vocab_size=vocab_size, hidden_size=self.hidden_size)
        self.sentence_embed2 = SentenceEmbedding(vocab_size=vocab_size, hidden_size=self.hidden_size)
        self.fc_embed = nn.Linear(args.fc_feat_size, self.hidden_size * 2)
        self.output_layer = nn.Linear(self.hidden_size * 4, 1)

    def forward(self, fc_feats, att_feats, att_masks, labels, seqs):
        device = fc_feats.device
        batch_size = labels.size(0)
        num_labels = labels.size(1)
        txt2txt = torch.zeros(num_labels, batch_size, self.hidden_size * 2, device=device)
        for i in range(num_labels):
            txt2txt[i] = self.sentence_embed1(labels[:, i]) * self.sentence_embed1(seqs)
        txt2txt = txt2txt.mean(0)
        im2txt = self._norm(self.fc_embed(fc_feats)) * self.sentence_embed2(seqs)
        outputs = self.output_layer(torch.cat([txt2txt, im2txt], 1)).squeeze(1)
        return outputs

    def _norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        y = (x - mean) / (std + 1e-8)
        return y
