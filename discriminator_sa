import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from attention import Attention


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        with open('data/vocab.json') as fid:
            vocab = json.load(fid)
        self.vocab_size = len(vocab) + 1
        self.max_length = 20
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size
        self.att_num = 3
        self.drop_prob = 0.2
        self.encoding_size = args.input_encoding_size
        self.bilstm = nn.LSTM(self.encoding_size,self.rnn_size,num_layers = 1 ,dropout = 0.1, bidirectional=True, bias=False, batch_first = True)
        self.embedding = nn.Embedding(self.vocab_size,args.input_encoding_size)
        self.embed_bilstm = nn.Sequential(nn.Linear(2*self.rnn_size, self.rnn_size),nn.Dropout(self.drop_prob))
        self.attention_lstm = nn.Sequential(nn.Linear(self.rnn_size,self.att_num),nn.Dropout(self.drop_prob))
        self.fc = nn.Linear(self.att_num * self.rnn_size*2, self.rnn_size,nn.Dropout(self.drop_prob))
        self.att_embed1 = nn.Sequential(nn.Linear(args.att_feat_size, args.rnn_size), nn.ReLU(),nn.Dropout(self.drop_prob))
        self.att_embed2 = nn.Sequential(nn.Linear(args.att_feat_size, args.att_hid_size),nn.Dropout(self.drop_prob))
        self.fc_embed = nn.Sequential(nn.Linear(args.fc_feat_size, args.rnn_size), nn.ReLU(),nn.Dropout(self.drop_prob))
        self.out_fc = nn.Sequential(nn.Linear(3*args.rnn_size,1))
        self.attention = Attention(args)

    def forward(self, fc_feats, att_feats, att_masks, seqs):
        bsz = att_feats.size(0)
        device = fc_feats.device
        # for bi-lstm and self-attention
        batch_sequence = self.embedding(seqs) # bsz * seq_length * encoding_size
        h_0 = torch.zeros(2, bsz, self.rnn_size, device=device)
        c_0 = torch.zeros(2, bsz, self.rnn_size, device=device)

        output,state = self.bilstm(batch_sequence,(h_0, c_0)) # bsz * seq_length * self.rnn_size*2

        tmp = self.embed_bilstm(output) # bsz * seq_length * self.rnn_size
        tanh_tmp = F.tanh(tmp)
        att_w = self.attention_lstm(tanh_tmp) # bsz * seq_length * self.att_num
        att_w = F.softmax(att_w,1)
        M = torch.bmm(att_w.transpose(1,2),output)  # bsz * self.att_num *  self.rnn_size*2
        output = F.relu(self.fc(M.view(bsz, -1)))

        # for image
        att_feats1 = self.att_embed1(att_feats)
        att_feats2 = self.att_embed2(att_feats)
        fc_feats = self.fc_embed(fc_feats)
        att = self.attention(output,att_feats1, att_feats2, att_masks)

        #concat and output prob
        output = torch.cat([output,fc_feats,att],1)
        probs = torch.sigmoid(self.out_fc(output))
        
        return probs
