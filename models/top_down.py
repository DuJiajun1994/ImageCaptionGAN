import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention


class TopDown(nn.Module):
    def __init__(self, args):
        super(TopDown, self).__init__()
        self.drop_prob = 0.5
        self.att_lstm = nn.LSTMCell(args.input_encoding_size + args.rnn_size * 2, args.rnn_size)
        self.lang_lstm = nn.LSTMCell(args.rnn_size * 2, args.rnn_size)
        self.attention = Attention(args)

    def forward(self, embed, fc_feats, att_feats1, att_feats2, att_masks, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, embed], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats1, att_feats2, att_masks)
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, state
