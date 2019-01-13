from generator import Generator
from discriminator import Discriminator
from evaluator import Evaluator
from loss import SequenceLoss, ReinforceLoss
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from coco_caption import CaptionDataset, DiscCaption, GeneratorCaption
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
        dataset = CaptionDataset('train', args)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        num_iter = 0
        for epoch in range(self.args.pretrain_generator_epochs):
            for data in loader:
                self.generator.train()
                for name, item in data.items():
                    data[name] = item.to(self.device)
                self.generator.zero_grad()
                probs = self.generator(data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'])
                loss = self.sequence_loss(probs, data['labels'])
                loss.backward()
                self._clip_gradient(self.generator_optimizer)
                self.generator_optimizer.step()
                print('iter {}, epoch {}, generator loss {:.3f}'.format(num_iter, epoch, loss.item()))
                num_iter += 1
            self.evaluator.evaluate_generator(self.generator)
            torch.save(self.generator.state_dict(), self.generator_checkpoint_path)

    def _pretrain_discriminator(self):
        num_iter = 0
        for epoch in range(self.args.pretrain_discriminator_epochs):
            for data in self.discriminator_loader:
                result = self._train_discriminator(data)
                print('iter {}, epoch {}, discriminator loss {:.3f}'.format(num_iter, epoch, result['loss']))
                print('real {:.3f}, wrong {:.3f}, fake {:.3f}'.format(result['real_prob'], result['wrong_prob'], result['fake_prob']))
                num_iter += 1
            self.evaluator.evaluate_discriminator(generator=self.generator, discriminator=self.discriminator)
            torch.save(self.discriminator.state_dict(), self.discriminator_checkpoint_path)
            
    def _train_gan(self):
        generator_dataset = GeneratorCaption('train', args)
        generator_loader = DataLoader(generator_dataset, batch_size=16, shuffle=True, num_workers=4)
        discriminator_iter = iter(self.discriminator_loader)
        num_iter = 0
        for epoch in range(self.args.train_gan_epochs):
            self._decay_learning_rate(epoch)
            for data in generator_loader:
                print('iter {}, epoch {}'.format(num_iter, epoch))
                num_iter += 1
                result = self._train_generator(data)
                print('generator loss {:.3f}'.format(result['loss']))
                print('cider {:.3f}'.format(result['cider_score']))
                try:
                    data = next(discriminator_iter)
                except StopIteration:
                    discriminator_iter = iter(self.discriminator_loader)
                    data = next(discriminator_iter)
                result = self._train_discriminator(data)
                print('discriminator loss {:.3f}'.format(result['loss']))
                print('real {:.3f}, wrong {:.3f}, fake {:.3f}'.format(result['real_prob'], result['wrong_prob'], result['fake_prob']))
            self.evaluator.evaluate_generator(self.generator)
            torch.save(self.generator.state_dict(), self.generator_checkpoint_path)
            self.evaluator.evaluate_discriminator(generator=self.generator, discriminator=self.discriminator)
            torch.save(self.discriminator.state_dict(), self.discriminator_checkpoint_path)

    def _train_generator(self, data):
        self.generator.train()
        self.discriminator.eval()
        for name, item in data.items():
            data[name] = item.to(self.device)

        self.generator.zero_grad()
        loss1 = self._xe_loss(data)
        loss1.backward()
        self._clip_gradient(self.generator_optimizer)
        self.generator_optimizer.step()

        self.generator.zero_grad()
        loss2, score = self._rl_loss(data)
        loss2.backward()
        self._clip_gradient(self.generator_optimizer)
        self.generator_optimizer.step()
        result = {
            'loss': loss1.item(),
            'cider_score': score
        }
        return result

    def _train_discriminator(self, data):
        self.generator.eval()
        self.discriminator.train()
        for name, item in data.items():
            data[name] = item.to(self.device)
        self.discriminator.zero_grad()

        real_probs = self.discriminator(data['labels'], data['match_labels'])
        wrong_probs = self.discriminator(data['labels'], data['wrong_labels'])

        # generate fake data
        with torch.no_grad():
            fake_seqs = self.generator.beam_search(data['fc_feats'], data['att_feats'], data['att_masks'])
        fake_seqs = fake_seqs.view(-1, fake_seqs.size(-1))
        labels = data['labels'].unsqueeze(1).expand(-1, 2, data['labels'].size(-1)).contiguous().view(-1, data['labels'].size(-1))
        fake_probs = self.discriminator(labels, fake_seqs)

        loss = -(0.5 * torch.log(real_probs + 1e-10).mean() + 0.25 * torch.log(1 - wrong_probs + 1e-10).mean() + 0.25 * torch.log(1 - fake_probs + 1e-10).mean())
        loss.backward()
        self._clip_gradient(self.discriminator_optimizer)
        self.discriminator_optimizer.step()
        result = {
            'loss': loss.item(),
            'real_prob': real_probs.mean().item(),
            'wrong_prob': wrong_probs.mean().item(),
            'fake_prob': fake_probs.mean().item()
        }
        return result

    def _xe_loss(self, data):
        batch_size = len(data['fc_feats'])
        fc_feats, att_feats, att_masks, images = self._expand(5, data['fc_feats'], data['att_feats'], data['att_masks'], data['images'])
        labels = data['labels'].transpose(0, 1).contiguous().view(5 * batch_size, -1)
        probs = self.generator(fc_feats, att_feats, att_masks, labels)
        loss = self.sequence_loss(probs, labels)
        return loss

    def _rl_loss(self, data):
        batch_size = len(data['fc_feats'])
        num_samples = self.args.beam_size
        with torch.no_grad():
            seqs = self.generator.beam_search(data['fc_feats'], data['att_feats'], data['att_masks'])
        seqs = seqs.transpose(0, 1).contiguous().view(num_samples * batch_size, -1)
        fc_feats, att_feats, att_masks, images = self._expand(num_samples, data['fc_feats'], data['att_feats'], data['att_masks'], data['images'])
        probs = self.generator(fc_feats, att_feats, att_masks, seqs)

        scores = self.cider.get_scores(seqs.cpu().numpy(), images.cpu().numpy())
        expand_seqs = seqs.unsqueeze(1).expand(num_samples * batch_size, 5, -1).contiguous().view(num_samples * batch_size * 5, -1)
        labels = data['labels']
        labels = labels.unsqueeze(0).expand(num_samples, batch_size, 5, -1).contiguous().view(num_samples * batch_size * 5, -1)
        with torch.no_grad():
            fake_probs = self.discriminator(labels, expand_seqs)
        fake_probs = fake_probs.view(num_samples * batch_size, 5).mean(1)
        reward = fake_probs
        baseline = reward.view(num_samples, batch_size).mean(0, keepdim=True).expand(num_samples, batch_size).contiguous().view(-1)
        loss = self.reinforce_loss(reward, baseline, probs, seqs)
        return loss, scores.mean()

    def _expand(self, num_samples, fc_feats, att_feats, att_masks, images):
        fc_feats = self._expand_tensor(num_samples, fc_feats)
        att_feats = self._expand_tensor(num_samples, att_feats)
        att_masks = self._expand_tensor(num_samples, att_masks)
        images = self._expand_tensor(num_samples, images)
        return fc_feats, att_feats, att_masks, images

    def _expand_tensor(self, num_samples, tensor):
        tensor_size = list(tensor.size())
        tensor = tensor.unsqueeze(0).expand([num_samples] + tensor_size)
        tensor_size[0] *= num_samples
        return tensor.contiguous().view(tensor_size)

    def _clip_gradient(self, optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-0.1, 0.1)

    def _decay_learning_rate(self, epoch):
        learning_rate_decay_rate = 0.8
        learning_rate_decay_every = 3
        if epoch % learning_rate_decay_every == 0:
            learning_rate = self.args.generator_lr * (learning_rate_decay_rate ** (epoch // learning_rate_decay_every))
            for group in self.generator_optimizer.param_groups:
                group['lr'] = learning_rate
            print('learning rate: {}'.format(learning_rate))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--generator_lr', type=float, default=5e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-4)
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
    parser.add_argument('--pretrain_generator_epochs', type=int, default=30)
    parser.add_argument('--pretrain_discriminator_epochs', type=int, default=30)
    parser.add_argument('--train_gan_epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    gan = GAN(device, args)
    gan.train()
