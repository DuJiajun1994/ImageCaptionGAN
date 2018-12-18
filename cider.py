import sys
sys.path.append("cider")
import pickle
from pyciderevalcap.ciderD.ciderD import CiderD


class Cider:
    def __init__(self, args):
        self.cider = CiderD(df='coco-train')
        with open('data/train_references.pkl') as fid:
            self.references = pickle.load(fid)

    def get_scores(self, seqs, images):
        captions = self._get_captions(seqs)
        res = [{'image_id': i, 'caption': [caption]}
               for i, caption in enumerate(captions)]
        gts = {i: self.references[image_id] for i, image_id in enumerate(images)}
        _, scores = self.cider.compute_score(gts, res)
        return scores

    def _get_captions(self, seqs):
        captions = [self._get_caption(seq) for seq in seqs]
        return captions

    def _get_caption(self, seq):
        words = []
        for word in seq:
            words.append(str(word))
            if word == 0:
                break
        caption = ' '.join(words)
        return caption
