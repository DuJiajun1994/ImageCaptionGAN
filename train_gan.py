from generator import Generator
from discriminator import Discriminator
from evaluate import evaluate
from loss import SequenceLoss
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_caption import TrainCaption, TestCaption


class GAN:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(self.device, args).to(self.device)
        self.discriminator = Discriminator(self.device, args).to(self.device)

    def train(self):
        self._pretrain_generator()
        self._pretrain_discriminator()
        self._train_gan()

    def _pretrain_generator(self):
        train_caption = TrainCaption(self.args)
        train_loader = DataLoader(train_caption, batch_size=16, shuffle=True, num_workers=4)
        val_caption = TestCaption('val', self.args)
        val_loader = DataLoader(val_caption, batch_size=16, shuffle=False, num_workers=4)
        sequence_loss = SequenceLoss()
        optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        iter = 0
        for epoch in range(25):
            self.generator.train()
            for data in train_loader:
                for _, item in data.items():
                    item.to(self.device)
                self.generator.zero_grad()
                prob = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
                loss = sequence_loss(prob, data['labels'], data['masks'])
                loss.backward()
                optimizer.step()
                print('iter {}, epoch {}, train loss {:.3f}'.format(iter, epoch, loss.item()))
                iter += 1
            evaluate(self.generator, val_loader)

    def _pretrain_discriminator(self):
        pass

    def _train_gan(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=1024)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gan = GAN(args)
    gan.train()
