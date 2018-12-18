import json


class Vocab:
    def __init__(self, args):
        with open('data/vocab.json') as fid:
            self.vocab = json.load(fid)

    def decode_captions(self, seqs):
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
