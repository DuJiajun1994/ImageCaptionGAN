import sys
sys.path.append('coco-caption')
import os
import json
import torch
from torch.utils.data import DataLoader
from coco_caption import TestCaption
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class Evaluator:
    def __init__(self, split, device, args):
        self.device = device
        self.annotation_file = 'coco-caption/annotations/captions_val2014.json'
        dataset = TestCaption(split, args)
        self.loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        with open('data/vocab.json') as fid:
            self.vocab = json.load(fid)

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
            captions = self._decode_captions(seqs)
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

    def _decode_captions(self, seqs):
        captions = []
        for seq in seqs:
            caption = ''
            for i, word_id in enumerate(seq):
                if word_id > 0:
                    if i == 0:
                        caption = self.vocab[str(word_id)]
                    else:
                        caption = caption + ' ' + self.vocab[str(word_id)]
                else:
                    break
            captions.append(caption)
        return captions
