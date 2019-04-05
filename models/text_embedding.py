import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.top_down import TopDown
from loss import get_masks


class TextEmbedding(nn.Module):
    def __init__(self, fc_feat_size, att_feat_size, args):
        super(TextEmbedding, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        self.vocab_size = len(vocab) + 1
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size

        self.embedding = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.fc_embed = nn.Sequential(nn.Linear(fc_feat_size, args.rnn_size), nn.ReLU(), nn.Dropout(0.5))
        self.att_embed1 = nn.Sequential(nn.Linear(att_feat_size, args.rnn_size), nn.ReLU(), nn.Dropout(0.5))
        self.att_embed2 = nn.Linear(att_feat_size, args.att_hid_size)
        self.decoder = TopDown(args)
        self.output_layer = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, fc_feats, att_feats, att_masks, seqs):
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        seq_length = seqs.size(1)
        outputs = torch.zeros(batch_size, seq_length, self.rnn_size, device=device)
        state = self._init_state(batch_size, device=device)
        for i in range(seq_length):
            outputs[:, i], state = self._core(seqs[:, i], fc_feats, att_feats1, att_feats2, att_masks, state)
        masks = get_masks(seqs)
        outputs = torch.sum(outputs * masks.unsqueeze(2).float(), dim=1)
        return outputs

    def _prepare_feature(self, fc_feats, att_feats):
        fc_feats = self.fc_embed(fc_feats)
        att_feats1 = self.att_embed1(att_feats)
        att_feats2 = self.att_embed2(att_feats)
        return fc_feats, att_feats1, att_feats2

    def _init_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.rnn_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.rnn_size, device=device))

    def _core(self, words, fc_feats, att_feats1, att_feats2, att_masks, state):
        embed = self.embedding(words)
        output, state = self.decoder(embed, fc_feats, att_feats1, att_feats2, att_masks, state)
        output = F.softmax(self.output_layer(output), dim=1)
        return output, state
