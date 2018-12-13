from generator import Generator
from discriminator import Discriminator
from evaluator import Evaluator
from loss import SequenceLoss
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_caption import TrainCaption


class GAN:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.device = device
        self.generator = Generator(args).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)

    def train(self):
        self._pretrain_generator()
        self._pretrain_discriminator()
        self._train_gan()

    def _pretrain_generator(self):
        train_caption = TrainCaption(self.args)
        train_loader = DataLoader(train_caption, batch_size=args.batch_size, shuffle=True, num_workers=4)
        evaluator = Evaluator('val', self.device, self.args)
        sequence_loss = SequenceLoss()
        optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        iter = 0
        for epoch in range(25):
            self.generator.train()
            for data in train_loader:
                for name, item in data.items():
                    data[name] = item.to(self.device)
                self.generator.zero_grad()
                prob = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
                loss = sequence_loss(prob, data['labels'], data['masks'])
                loss.backward()
                optimizer.step()
                print('iter {}, epoch {}, train loss {:.3f}'.format(iter, epoch, loss.item()))
                iter += 1
            evaluator.evaluate(self.generator)

    def _pretrain_discriminator(self):
        pass

    def _train_gan(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gan = GAN(device, args)
    gan.train()
