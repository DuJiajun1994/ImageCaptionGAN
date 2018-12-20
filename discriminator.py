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
        #self.vocab_size = 1000
        self.hidden_size = args.d_rnn_size
        self.input_size = args.disc_input_size
        self.batch_size = args.batch_size
        self.fc_feat_size = args.fc_feat_size
        self.dis_lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers = 1, batch_first = True)
        #embedding for visual input
        self.vembedding = nn.Linear(self.fc_feat_size,self.input_size)

        #word embedding
        self.wembedding = nn.Embedding(self.vocab_size, self.input_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)
    def forward(self, fc_feats, att_feats, att_masks, seqs):
        """

        :param fc_feats:bs x fc_size
        :param att_feats:...
        :param att_masks:...
        :param seqs:bs x max_seqlens
        :return: probabilities of the captions being real
        """
        device = fc_feats.device
        #initial state for h_0,c_0. num_layer, batch_size, hidden_size
        batch_size = fc_feats.size(0)
        h_0 = torch.zeros(1,batch_size, self.hidden_size,device=device)
        c_0 = torch.zeros(1,batch_size, self.hidden_size,device=device)
        x_0 = fc_feats#self.vembedding(fc_feats)
        emb_x = self.wembedding(seqs)
        #add image to be the first input of the seq
        new_x = torch.cat((x_0.unsqueeze(1),emb_x),dim=1)
        #get mask
        mask = (seqs > 0).sum(1)
        #print (mask)
        output,(h_n,c_n) = self.dis_lstm(new_x, (h_0, c_0))
        #print (output)
        valid_output = [output[b,s,:] for b,s in enumerate(mask)]
        valid_output = torch.cat(valid_output).view(batch_size,-1)
        #print (valid_output)
        out = self.output_layer(valid_output)
        out = torch.sigmoid(out).squeeze(1)
        return out

