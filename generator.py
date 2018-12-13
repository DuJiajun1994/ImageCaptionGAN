import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.top_down import TopDown


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        self.vocab_size = len(vocab)
        self.max_length = 20
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size

        self.embedding = nn.Embedding(self.vocab_size + 1, args.input_encoding_size)
        self.fc_embed = nn.Sequential(nn.Linear(args.fc_feat_size, args.rnn_size), nn.ReLU())
        self.att_embed1 = nn.Sequential(nn.Linear(args.att_feat_size, args.rnn_size), nn.ReLU())
        self.att_embed2 = nn.Linear(args.att_feat_size, args.att_hid_size)
        self.decoder = TopDown(args)
        self.output_layer = nn.Linear(args.rnn_size, self.vocab_size + 1)

    def forward(self, fc_feats, att_feats, att_masks, labels):
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        seq_length = labels.size(1)
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size + 1, device=device)
        state = self._init_state(batch_size, device=device)
        for i in range(seq_length):
            if i == 0:
                input = torch.zeros(batch_size, dtype=torch.long, device=device)  # BOS
            else:
                input = labels[:, i-1]
            output, state = self._core(input, fc_feats, att_feats1, att_feats2, att_masks, state)
            outputs[:, i] = output
            if labels[:, i].sum() == 0:
                break
        return outputs

    def beam_search(self, fc_feats, att_feats, att_masks):
        pass

    def sample(self, fc_feats, att_feats, att_masks):
        pass

    def greedy_decode(self, fc_feats, att_feats, att_masks):
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        tokens = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        token = torch.zeros(batch_size, dtype=torch.long, device=device)    # BOS
        state = self._init_state(batch_size, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)
        for i in range(self.max_length):
            output, state = self._core(token, fc_feats, att_feats1, att_feats2, att_masks, state)
            _, token = torch.max(output, dim=1)
            token = token * unfinished
            unfinished = unfinished * (token > 0).long()
            tokens[:, i] = token
            if unfinished.sum() == 0:
                break
        return tokens

    def _prepare_feature(self, fc_feats, att_feats):
        att_feats = att_feats.view(att_feats.size(0), -1, att_feats.size(-1))
        fc_feats = self.fc_embed(fc_feats)
        att_feats1 = self.att_embed1(att_feats)
        att_feats2 = self.att_embed2(att_feats)
        return fc_feats, att_feats1, att_feats2

    def _init_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.rnn_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.rnn_size, device=device))

    def _core(self, input, fc_feats, att_feats1, att_feats2, att_masks, state):
        embed = self.embedding(input)
        output, state = self.decoder(embed, fc_feats, att_feats1, att_feats2, att_masks, state)
        output = F.softmax(self.output_layer(output), dim=1)
        return output, state
