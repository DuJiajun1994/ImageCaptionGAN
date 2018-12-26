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
from coco_caption import CaptionDataset, DiscCaption
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
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=args.generator_lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.discriminator_lr)
        self.evaluator = Evaluator('val', self.device, args)
        self.cider = Cider(args)
        generator_dataset = CaptionDataset('train', args)
        self.generator_loader = DataLoader(generator_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        discriminator_dataset = DiscCaption('train', args)
        self.discriminator_loader = DataLoader(discriminator_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def train(self):
        if self.args.load_generator:
            self.generator.load_state_dict(torch.load(self.generator_checkpoint_path))
        else:
            self._pretrain_generator()
        if self.args.load_discriminator:
            self.discriminator.load_state_dict(torch.load(self.discriminator_checkpoint_path))
        else:
            self._pretrain_discriminator()
        self._train_gan()

    def _pretrain_generator(self):
        iter = 0
        for epoch in range(self.args.pretrain_generator_epochs):
            self.generator.train()
            for data in self.generator_loader:
                for name, item in data.items():
                    data[name] = item.to(self.device)
                self.generator.zero_grad()
                probs = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
                loss = self.sequence_loss(probs, data['labels'])
                loss.backward()
                self.generator_optimizer.step()
                print('iter {}, epoch {}, generator loss {:.3f}'.format(iter, epoch, loss.item()))
                iter += 1
            self.evaluator.evaluate_generator(self.generator)
            torch.save(self.generator.state_dict(), self.generator_checkpoint_path)

    def _pretrain_discriminator(self):
        iter = 0
        for epoch in range(self.args.pretrain_discriminator_epochs):
            self.discriminator.train()
            for data in self.discriminator_loader:
                loss = self._train_discriminator(data)
                print('iter {}, epoch {}, discriminator loss {:.3f}'.format(iter, epoch, loss))
                iter += 1
            self.evaluator.evaluate_discriminator(generator=self.generator, discriminator=self.discriminator)
            torch.save(self.discriminator.state_dict(), self.discriminator_checkpoint_path)
            
    def _train_gan(self):
        generator_iter = iter(self.generator_loader)
        discriminator_iter = iter(self.discriminator_loader)
        for i in range(self.args.train_gan_iters):
            print('iter {}'.format(i))
            for j in range(1):
                try:
                    data = next(generator_iter)
                except StopIteration:
                    generator_iter = iter(self.generator_loader)
                    data = next(generator_iter)
                result = self._train_generator(data)
                print('generator loss {:.3f}, fake prob {:.3f}, cider score {:.3f}'.format(result['loss'], result['fake_prob'], result['cider_score']))
            for j in range(1):
                try:
                    data = next(discriminator_iter)
                except StopIteration:
                    discriminator_iter = iter(self.discriminator_loader)
                    data = next(discriminator_iter)
                loss = self._train_discriminator(data)
                print('discriminator loss {:.3f}'.format(loss))
            if i != 0 and i % 10000 == 0:
                self.evaluator.evaluate_generator(self.generator)
                torch.save(self.generator.state_dict(), self.generator_checkpoint_path)
                self.evaluator.evaluate_discriminator(generator=self.generator, discriminator=self.discriminator)
                torch.save(self.discriminator.state_dict(), self.discriminator_checkpoint_path)

    def _train_generator(self, data):
        self.generator.train()
        for name, item in data.items():
            data[name] = item.to(self.device)
        self.generator.zero_grad()

        probs = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
        loss1 = self.sequence_loss(probs, data['labels'])

        seqs, probs = self.generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
        greedy_seqs = self.generator.greedy_decode(data['fc_feats'], data['att_feats'], data['att_masks'])
        reward, fake_prob, score = self._get_reward(data, seqs)
        baseline, _, _ = self._get_reward(data, greedy_seqs)
        loss2 = self.reinforce_loss(reward, baseline, probs, seqs)

        loss = loss1 + loss2
        loss.backward()
        self.generator_optimizer.step()
        result = {
            'loss': loss1.item(),
            'fake_prob': fake_prob,
            'cider_score': score
        }
        return result

    def _train_discriminator(self, data):
        self.discriminator.train()
        for name, item in data.items():
            data[name] = item.to(self.device)
        self.discriminator.zero_grad()

        real_probs = self.discriminator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
        wrong_probs = self.discriminator(data['fc_feats'], data['att_feats'], data['att_masks'], data['wrong_labels'])

        # generate fake data
        with torch.no_grad():
            fake_seqs, _ = self.generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
        fake_probs = self.discriminator(data['fc_feats'], data['att_feats'], data['att_masks'], fake_seqs)

        loss = -(0.5 * torch.log(real_probs + 1e-10) + 0.25 * torch.log(1 - wrong_probs + 1e-10) + 0.25 * torch.log(1 - fake_probs + 1e-10)).mean()
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _get_reward(self, data, seqs):
        probs = self.discriminator(data['fc_feats'], data['att_feats'], data['att_masks'], seqs)
        scores = self.cider.get_scores(seqs.cpu().numpy(), data['images'].cpu().numpy())
        reward = probs + torch.tensor(scores, dtype=torch.float, device=self.device)
        fake_prob = probs.mean().item()
        score = scores.mean()
        return reward, fake_prob, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--generator_lr', type=float, default=1e-3)
    parser.add_argument('--discriminator_lr', type=float, default=1e-3)
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--d_rnn_size', type=int, default=512)
    parser.add_argument('--disc_input_size', type=int, default=2048)
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att')
    parser.add_argument('--checkpoint_path', type=str, default='output')
    parser.add_argument('--load_generator', type=int, default=0)
    parser.add_argument('--load_discriminator', type=int, default=0)
    parser.add_argument('--pretrain_generator_epochs', type=int, default=10)
    parser.add_argument('--pretrain_discriminator_epochs', type=int, default=10)
    parser.add_argument('--train_gan_iters', type=int, default=1000000)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    gan = GAN(device, args)
    gan.train()
