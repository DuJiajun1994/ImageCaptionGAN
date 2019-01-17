import sys
sys.path.append('coco-caption')
import os
import json
import torch
import argparse
import numpy as np
from scipy.stats.stats import pearsonr
from torch.utils.data import DataLoader
from coco_caption import ImageDataset, DiscCaption, CaptionDataset
from generator import Generator
from discriminator import Discriminator
from vocab import Vocab
from cider import Cider
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class Evaluator:
    def __init__(self, split, device, args):
        self.device = device
        self.annotation_file = 'coco-caption/annotations/captions_val2014.json'
        generator_dataset = ImageDataset(split=split, expand_by_labels=False, args=args)
        self.generator_loader = DataLoader(generator_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        discriminator_dataset = DiscCaption('val', args)
        self.discriminator_loader = DataLoader(discriminator_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        correlation_dataset = CaptionDataset('train', args)
        self.correlation_loader = DataLoader(correlation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.vocab = Vocab(args)
        self.cider = Cider(args)

    def evaluate_generator(self, generator):
        predictions = self._generate_predictions(generator)
        metrics = self._evaluate_predictions(predictions)
        return metrics

    def evaluate_discriminator(self, generator, discriminator):
        generator.eval()
        discriminator.eval()
        num_total = 0
        sum_real = 0
        sum_wrong = 0
        sum_fake = 0
        for data in self.discriminator_loader:
            for name, item in data.items():
                data[name] = item.to(self.device)
            with torch.no_grad():
                real_probs = discriminator(data['labels'], data['match_labels'])
                wrong_probs = discriminator(data['labels'], data['wrong_labels'])
                fake_seqs, _ = generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
                fake_probs = discriminator(data['labels'], fake_seqs)
            num_total += len(real_probs)
            sum_real += real_probs.sum().item()
            sum_wrong += wrong_probs.sum().item()
            sum_fake += fake_probs.sum().item()
        real_prob = sum_real / float(num_total)
        wrong_prob = sum_wrong / float(num_total)
        fake_prob = sum_fake / float(num_total)
        print('real prob: {}'.format(real_prob))
        print('wrong prob: {}'.format(wrong_prob))
        print('fake prob: {}'.format(fake_prob))

    def evaluate_correlation(self, generator, discriminator):
        generator.eval()
        discriminator.eval()
        num_sample_total = 32
        corrs = []
        cnt = 0
        for data in self.correlation_loader:
            cnt += 1
            if cnt > 1000:
                break
            score = []
            fake = []
            for name, item in data.items():
                data[name] = item.to(self.device)
            for i in range(num_sample_total):
                with torch.no_grad():
                    fake_seqs, probs = generator.sample(data['fc_feats'], data['att_feats'], data['att_masks'])
                    fake_probs = discriminator(data['labels'], fake_seqs)
                scores = self.cider.get_scores(fake_seqs.cpu().numpy(), data['images'].cpu().numpy())
                score.append(scores)
                fake.append(fake_probs.cpu().numpy())
            score = np.array(score)
            fake = np.array(fake)
            for j in range(score.shape[1]):
                corr, _ = pearsonr(score[:, j], fake[:, j])
                corrs.append(corr)
                print(corr)
        print('correlation mean: {}'.format(np.array(corrs).mean()))

    def _generate_predictions(self, generator):
        generator.eval()
        predictions = []
        for data in self.generator_loader:
            images = data['images'].cpu().numpy()
            for name, item in data.items():
                data[name] = item.to(self.device)
            with torch.no_grad():
                seqs = generator.beam_search(data['fc_feats'], data['att_feats'], data['att_masks'])
            captions = self.vocab.decode_captions(seqs.cpu().numpy())
            for i, caption in enumerate(captions):
                image_id = images[i]
                predictions.append({
                    'image_id': image_id,
                    'caption': caption
                })
                print('{} {}'.format(image_id, caption))
        return predictions

    def _evaluate_predictions(self, predictions):
        if not os.path.isdir('eval_results'):
            os.mkdir('eval_results')
        cache_path = os.path.join('eval_results/predictions.json')

        coco = COCO(self.annotation_file)
        with open(cache_path, 'w') as fid:
            json.dump(predictions, fid)

        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()

        metrics = {}
        for metric, score in cocoEval.eval.items():
            metrics[metric] = score
        return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
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
    parser.add_argument('--evaluate_generator', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(args.split, device, args)
    generator = Generator(args).to(device)
    state_dict = torch.load(os.path.join(args.checkpoint_path, 'generator.pth'))
    generator.load_state_dict(state_dict)
    if args.evaluate_generator == 1:
        evaluator.evaluate_generator(generator)
    else:
        discriminator = Discriminator(args).to(device)
        state_dict = torch.load(os.path.join(args.checkpoint_path, 'discriminator.pth'))
        discriminator.load_state_dict(state_dict)
        if args.evaluate_generator == 0:
            evaluator.evaluate_discriminator(generator, discriminator)
        elif args.evaluate_generator == 2:
            evaluator.evaluate_correlation(generator, discriminator)
