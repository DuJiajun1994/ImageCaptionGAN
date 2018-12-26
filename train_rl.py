from generator import Generator
from evaluator import Evaluator
from loss import SequenceLoss, ReinforceLoss
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_caption import CaptionDataset
from cider import Cider


class RL:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.batch_size = args.batch_size
        self.checkpoint_path = os.path.join(args.checkpoint_dir, 'generator.pth')
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        self.generator = Generator(args).to(self.device)
        self.sequence_loss = SequenceLoss()
        self.reinforce_loss = ReinforceLoss()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=args.learning_rate)
        self.evaluator = Evaluator('val', self.device, args)
        self.cider = Cider(args)
        dataset = CaptionDataset('train', args)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def train(self):
        if self.args.load_generator:
            self.generator.load_state_dict(torch.load(self.checkpoint_path))
        else:
            self._train_xe()
        self._train_rl()

    def _train_xe(self):
        iter = 0
        for epoch in range(self.args.xe_epochs):
            self._decay_learning_rate(epoch)
            self.generator.train()
            for data in self.loader:
                for name, item in data.items():
                    data[name] = item.to(self.device)
                self.generator.zero_grad()
                probs = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
                loss = self.sequence_loss(probs, data['labels'])
                loss.backward()
                self._clip_gradient()
                self.optimizer.step()
                print('iter {}, epoch {}, loss {:.3f}'.format(iter, epoch, loss.item()))
                iter += 1
            self.evaluator.evaluate_generator(self.generator)
            torch.save(self.generator.state_dict(), self.checkpoint_path)

    def _train_rl(self):
        iter = 0
        for epoch in range(self.args.xe_epochs, self.args.xe_epochs + self.args.rl_epochs):
            self._decay_learning_rate(epoch)
            for data in self.loader:
                self.generator.train()
                for name, item in data.items():
                    data[name] = item.to(self.device)
                self.generator.zero_grad()
                loss, reward = self._rl_core1(data)
                loss.backward()
                self._clip_gradient()
                self.optimizer.step()
                print('iter {}, epoch {}, cider score {:.3f}'.format(iter, epoch, reward.mean().item()))
                iter += 1
            self.evaluator.evaluate_generator(self.generator)
            torch.save(self.generator.state_dict(), self.checkpoint_path)

    def _get_reward(self, data, seqs):
        scores = self.cider.get_scores(seqs.cpu().numpy(), data['images'].cpu().numpy())
        reward = torch.tensor(scores, dtype=torch.float, device=self.device)
        return reward

    def _clip_gradient(self):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-self.args.grad_clip_threshold, self.args.grad_clip_threshold)

    def _decay_learning_rate(self, epoch):
        if epoch % self.args.learning_rate_decay_every == 0:
            learning_rate = self.args.learning_rate * (self.args.learning_rate_decay_rate ** (epoch // self.args.learning_rate_decay_every))
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate
            print('learning rate: {}'.format(learning_rate))

    def _rl_core1(self, data):
        seqs, probs = self.generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
        greedy_seqs = self.generator.greedy_decode(data['fc_feats'], data['att_feats'], data['att_masks'])
        reward = self._get_reward(data, seqs)
        baseline = self._get_reward(data, greedy_seqs)
        loss = self.reinforce_loss(reward, baseline, probs, seqs)
        return loss, reward

    def _rl_core2(self, data):
        num_samples = 8
        all_seqs = []
        all_probs = []
        all_reward = []
        for _ in range(num_samples):
            seqs, probs = self.generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
            scores = self.cider.get_scores(seqs.cpu().numpy(), data['images'].cpu().numpy())
            reward = torch.tensor(scores, dtype=torch.float, device=self.device)
            all_seqs.append(seqs)
            all_probs.append(probs)
            all_reward.append(reward)
        all_seqs = torch.stack(all_seqs)
        all_probs = torch.stack(all_probs)
        all_reward = torch.stack(all_reward)
        seqs = all_seqs.view(-1, all_seqs.size(2))
        probs = all_probs.view(-1, all_probs.size(2), all_probs.size(3))
        baseline = all_reward.mean(0, keepdim=True).expand(num_samples, -1).contiguous().view(-1)
        reward = all_reward.view(-1)
        loss = self.reinforce_loss(reward, baseline, probs, seqs)
        return loss, reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--grad_clip_threshold', type=float, default=0.1)
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att')
    parser.add_argument('--checkpoint_dir', type=str, default='output')
    parser.add_argument('--load_generator', type=int, default=0)
    parser.add_argument('--xe_epochs', type=int, default=30)
    parser.add_argument('--rl_epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    rl = RL(device, args)
    rl.train()
