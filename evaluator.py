import sys
sys.path.append('coco-caption')
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from coco_caption import ImageDataset
from generator import Generator
from vocab import Vocab
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class Evaluator:
    def __init__(self, split, device, args):
        self.device = device
        self.annotation_file = 'coco-caption/annotations/captions_val2014.json'
        dataset = ImageDataset(split, args)
        self.loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.vocab = Vocab(args)

    def evaluate(self, generator):
        predictions = self._generate_predictions(generator)
        metrics = self._evaluate_predictions(predictions)
        return metrics

    def _generate_predictions(self, generator):
        generator.eval()
        predictions = []
        for data in self.loader:
            images = data['images'].cpu().numpy()
            for name, item in data.items():
                data[name] = item.to(self.device)
            with torch.no_grad():
                seqs = generator.beam_search(data['fc_feats'], data['att_feats'], data['att_masks']).cpu().numpy()
            captions = self.vocab.decode_captions(seqs)
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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(args.split, device, args)
    generator = Generator(args).to(device)
    state_dict = torch.load(os.path.join(args.checkpoint_path, 'generator.pth'))
    generator.load_state_dict(state_dict)
    evaluator.evaluate(generator)
