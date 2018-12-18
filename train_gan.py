from generator import Generator
from discriminator import Discriminator
from evaluator import Evaluator
from loss import SequenceLoss, ReinforceLoss
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_caption import TrainCaption
from cider import Cider


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
        self.sequence_loss = SequenceLoss()
        self.reinforce_loss = ReinforceLoss()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.evaluator = Evaluator('val', self.device, self.args)
        self.cider = Cider(args)

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
        iter = 0
        for epoch in range(25):
            self.generator.train()
            for data in train_loader:
                for name, item in data.items():
                    data[name] = item.to(self.device)
                self.generator.zero_grad()
                probs = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
                loss = self.sequence_loss(probs, data['labels'])
                loss.backward()
                self.generator_optimizer.step()
                print('iter {}, epoch {}, train loss {:.3f}'.format(iter, epoch, loss.item()))
                iter += 1
            self.evaluator.evaluate(self.generator)
            torch.save(self.generator.state_dict(), self.generator_checkpoint_path)

    def _pretrain_discriminator(self):
        pass

    def _train_gan(self):
        train_caption = TrainCaption(self.args)
        loader1 = DataLoader(train_caption, batch_size=self.batch_size, shuffle=True, num_workers=4)
        iter1 = iter(loader1)
        for i in range(100000):
            for j in range(1):
                try:
                    data = next(iter1)
                except StopIteration:
                    iter1 = iter(loader1)
                    data = next(iter1)
                self._train_generator(data)
            for j in range(1):
                self._train_discriminator()
            if i != 0 and i % 10000 == 0:
                self.evaluator.evaluate(self.generator)
                torch.save(self.generator.state_dict(), self.generator_checkpoint_path)

    def _train_generator(self, data):
        for name, item in data.items():
            data[name] = item.to(self.device)
        self.generator.zero_grad()

        probs = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
        loss1 = self.sequence_loss(probs, data['labels'])

        seqs, probs = self.generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
        greedy_seqs = self.generator.beam_search(data['fc_feats'], data['att_feats'], data['att_masks'])
        reward = self._get_reward(data, seqs)
        baseline = self._get_reward(data, greedy_seqs)
        loss2 = self.reinforce_loss(reward, baseline, probs, seqs)

        loss = loss1 + loss2
        loss.backward()
        self.generator_optimizer.step()

    def _train_discriminator(self):
        pass

    def _get_reward(self, data, seqs):
        probs = self.discriminator(data['fc_feats'], data['att_feats'], data['att_masks'], seqs)
        scores = self.cider.get_scores(seqs, data['images'])
        reward = probs + scores
        return reward


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
