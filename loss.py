import torch
import torch.nn as nn


def get_masks(seqs):
    seq_len = (seqs > 0).sum(dim=1) + 1
    masks = torch.zeros_like(seqs, dtype=torch.uint8)
    for i in range(len(seqs)):
        masks[i, :seq_len[i]] = 1
    return masks


class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()

    def forward(self, probs, seqs):
        masks = get_masks(seqs)
        probs = probs.gather(2, seqs.unsqueeze(2)).squeeze(2)
        losses = - torch.log(probs + 1e-10)
        loss = losses.masked_select(masks).mean()
        return loss


class ReinforceLoss(nn.Module):
    def __init__(self):
        super(ReinforceLoss, self).__init__()

    def forward(self, reward, baseline, probs, seqs):
        masks = get_masks(seqs)
        probs = probs.gather(2, seqs.unsqueeze(2)).squeeze(2)
        losses = - torch.log(probs + 1e-10) * (reward.detach() - baseline.detach()).unsqueeze(1)
        loss = losses.masked_select(masks).mean()
        return loss
