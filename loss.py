import torch
import torch.nn as nn


class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()

    def forward(self, input, target, mask):
        prob = input.gather(2, target.unsqueeze(2)).squeeze(2)
        cross_entropy = - torch.log(prob)
        loss = cross_entropy.masked_select(mask).mean()
        return loss
