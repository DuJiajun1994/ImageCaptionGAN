import json
import torch
import torch.nn as nn
from models.text_embedding import TextEmbedding


class Txt2Txt(nn.Module):
    def __init__(self, args):
        super(Txt2Txt, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        vocab_size = len(vocab) + 1
        self.embedding = nn.Embedding(vocab_size, args.rnn_size)
        self.lstm = nn.LSTM(args.rnn_size, args.rnn_size, num_layers=1, batch_first=True)
        self.text_embed = TextEmbedding(fc_feat_size=args.rnn_size, att_feat_size=args.rnn_size, args=args)

    def forward(self, labels, seqs):
        embed = self.embedding(labels)
        att_feats, _ = self.lstm(embed)
        att_masks = (labels > 0).float()
        fc_feats = torch.sum(att_feats * att_masks.unsqueeze(2), dim=1) / (att_masks.sum(dim=1, keepdim=True) + 1e-10)
        outputs = self.text_embed(fc_feats, att_feats, att_masks, seqs)
        return outputs
