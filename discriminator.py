import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
