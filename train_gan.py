from generator import Generator
from discriminator import Discriminator
from evaluator import Evaluator
from loss import SequenceLoss
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_caption import TrainCaption


class GAN:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.batch_size = args.batch_size
        self.generator_checkpoint_path = os.path.join(args.checkpoint_path, 'generator.pth')
        self.discriminator_checkpoint_path = os.path.join(args.checkpoint_path, 'discriminator.pth')
        if not os.path.isdir(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
        self.generator = Generator(args).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)

    def train(self):
        if self.args.load_generator:
            state_dict = torch.load(self.generator_checkpoint_path)
            self.generator.load_state_dict(state_dict)
        else:
            self._pretrain_generator()
        if self.args.load_discriminator:
            state_dict = torch.load(self.discriminator_checkpoint_path)
            self.generator.load_state_dict(state_dict)
        else:
            self._pretrain_discriminator()
        self._train_gan()

    def _pretrain_generator(self):
        train_caption = TrainCaption(self.args)
        train_loader = DataLoader(train_caption, batch_size=self.batch_size, shuffle=True, num_workers=4)
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
            torch.save(self.generator.state_dict(), self.generator_checkpoint_path)

    def _pretrain_discriminator(self):
        pass

    def _train_gan(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att')
    parser.add_argument('--checkpoint_path', type=str, default='output')
    parser.add_argument('--load_generator', type=bool, default=False)
    parser.add_argument('--load_discriminator', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gan = GAN(device, args)
    gan.train()
