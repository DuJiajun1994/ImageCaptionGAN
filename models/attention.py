import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.h2att = nn.Linear(args.rnn_size, args.att_hid_size)
        self.alpha_net = nn.Linear(args.att_hid_size, 1)

    def forward(self, h, att_feats1, att_feats2, att_masks):
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att_feats2)
        dot = self.alpha_net(F.tanh(att_feats2 + att_h)).squeeze(2)
        weight = F.softmax(dot, dim=1)
        if att_masks is not None:
            weight = weight * att_masks.float()
            weight = weight / weight.sum(1, keepdim=True)
        att = torch.bmm(weight.unsqueeze(1), att_feats1).squeeze(1)
        return att
