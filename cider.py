import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
import torch


class Cider:
    def __init__(self, args):
        self.cider = CiderD(df='coco-train')

    def get_scores(self, seqs, labels):
        device = seqs.device
        seqs = seqs.cpu().numpy()
        labels = labels.cpu().numpy()
        res = [{'image_id': i, 'caption': [self._get_caption(seq)]} for i, seq in enumerate(seqs)]
        gts = {i: [self._get_caption(seq) for seq in label] for i, label in enumerate(labels)}
        _, scores = self.cider.compute_score(gts, res)
        return torch.tensor(scores, dtype=torch.float, device=device)

    def _get_caption(self, seq):
        words = []
        for word in seq:
            words.append(str(word))
            if word == 0:
                break
        caption = ' '.join(words)
        return caption
