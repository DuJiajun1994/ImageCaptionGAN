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
        self.vocab_size = len(vocab) + 1
        self.max_length = 20
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size
        self.beam_size = args.beam_size

        self.embedding = nn.Embedding(self.vocab_size, args.input_encoding_size)
        self.fc_embed = nn.Sequential(nn.Linear(args.fc_feat_size, args.rnn_size), nn.ReLU(), nn.Dropout(0.5))
        self.att_embed1 = nn.Sequential(nn.Linear(args.att_feat_size, args.rnn_size), nn.ReLU(), nn.Dropout(0.5))
        self.att_embed2 = nn.Linear(args.att_feat_size, args.att_hid_size)
        self.decoder = TopDown(args)
        self.output_layer = nn.Linear(args.rnn_size, self.vocab_size)

    def forward(self, fc_feats, att_feats, att_masks, labels):
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        seq_length = labels.size(1)
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size, device=device)
        state = self._init_state(batch_size, device=device)
        for i in range(seq_length):
            if i == 0:
                words = torch.zeros(batch_size, dtype=torch.long, device=device)  # BOS
            else:
                words = labels[:, i-1]
            output, state = self._core(words, fc_feats, att_feats1, att_feats2, att_masks, state)
            outputs[:, i] = output
            if labels[:, i].sum() == 0:
                break
        return outputs

    def sample(self, fc_feats, att_feats, att_masks):
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        seqs = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        probs = torch.zeros(batch_size, self.max_length, self.vocab_size, device=device)
        words = torch.zeros(batch_size, dtype=torch.long, device=device)  # BOS
        state = self._init_state(batch_size, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)
        for i in range(self.max_length):
            output, state = self._core(words, fc_feats, att_feats1, att_feats2, att_masks, state)
            words = output.multinomial(num_samples=1).squeeze(1) * unfinished
            unfinished = unfinished * (words > 0).long()
            seqs[:, i] = words
            probs[:, i] = output
            if unfinished.sum() == 0:
                break
        return seqs, probs

    def greedy_decode(self, fc_feats, att_feats, att_masks):
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        seqs = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        words = torch.zeros(batch_size, dtype=torch.long, device=device)    # BOS
        state = self._init_state(batch_size, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)
        for i in range(self.max_length):
            output, state = self._core(words, fc_feats, att_feats1, att_feats2, att_masks, state)
            words = output.argmax(dim=1) * unfinished
            unfinished = unfinished * (words > 0).long()
            seqs[:, i] = words
            if unfinished.sum() == 0:
                break
        return seqs

    def beam_search(self, fc_feats, att_feats, att_masks):
        device = fc_feats.device
        batch_size = fc_feats.size(0)
        seqs = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        for i in range(batch_size):
            seqs[i] = self._beam_search_single_sample(fc_feats[i], att_feats[i], att_masks[i])
        return seqs

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

    def _beam_search_single_sample(self, fc_feats, att_feats, att_masks):
        device = fc_feats.device
        beam_size = self.beam_size
        fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
        fc_feats, att_feats1, att_feats2, att_masks = self._expand(beam_size, fc_feats, att_feats1, att_feats2, att_masks)

        seqs = torch.zeros(beam_size, self.max_length, dtype=torch.long, device=device)
        words = torch.zeros(beam_size, dtype=torch.long, device=device)  # BOS
        state = self._init_state(beam_size, device=device)
        output, state = self._core(words, fc_feats, att_feats1, att_feats2, att_masks, state)
        word_scores = torch.log(output + 1e-10)
        scores, words = word_scores[0].topk(beam_size, largest=True, sorted=True)
        seqs[:, 0] = words
        for i in range(1, self.max_length):
            unfinished = (words > 0).long()
            if unfinished.sum() == 0:
                break
            output, state = self._core(words, fc_feats, att_feats1, att_feats2, att_masks, state)
            word_scores = torch.log(output + 1e-10)
            word_scores[:, word_scores.size(1) - 1] = word_scores[:, word_scores.size(1) - 1] - 1000    # suppress UNK
            unfinish_idx = unfinished.nonzero()
            finish_idx = (1 - unfinished).nonzero()
            unfinish_scores = (scores[unfinish_idx].unsqueeze(1).expand_as(word_scores[unfinish_idx]) + word_scores[unfinish_idx]).view(-1)
            finish_scores = scores[finish_idx].view(-1)
            num_unfinish = unfinish_scores.size(0)
            scores = torch.cat([unfinish_scores, finish_scores])
            scores, idx = scores.topk(beam_size, largest=True, sorted=True)
            seq_idx = torch.zeros(beam_size, dtype=torch.long, device=device)
            for j in range(beam_size):
                if idx[j] < num_unfinish:
                    seq_idx[j] = unfinish_idx[idx[j] / self.vocab_size]
                else:
                    seq_idx[j] = finish_idx[idx[j] - num_unfinish]
            words = (idx % self.vocab_size) * (idx < num_unfinish).long()
            seqs = seqs[seq_idx]
            seqs[:, i] = words
            state = self._choose_state(state, seq_idx, device)
        return seqs[0]

    def _expand(self, expand_size, fc_feats, att_feats1, att_feats2, att_masks):
        fc_feats = self._expand_tensor(expand_size, fc_feats)
        att_feats1 = self._expand_tensor(expand_size, att_feats1)
        att_feats2 = self._expand_tensor(expand_size, att_feats2)
        att_masks = self._expand_tensor(expand_size, att_masks)
        return fc_feats, att_feats1, att_feats2, att_masks

    def _expand_tensor(self, expand_size, tensor):
        tensor = tensor.unsqueeze(0)
        tensor_size = list(tensor.size())
        tensor_size[0] = expand_size
        tensor = tensor.expand(tensor_size)
        return tensor

    def _choose_state(self, state, idx, device):
        batch_size = idx.size(0)
        new_state = (torch.zeros(self.num_layers, batch_size, self.rnn_size, device=device),
                     torch.zeros(self.num_layers, batch_size, self.rnn_size, device=device))
        for i in range(batch_size):
            new_state[0][:, i, :] = state[0][:, idx[i], :]
            new_state[1][:, i, :] = state[1][:, idx[i], :]
        return new_state
