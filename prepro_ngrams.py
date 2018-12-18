import json
import argparse
import pickle
from collections import defaultdict


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.iteritems()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def build_dict(imgs, wtoi, args):
    wtoi['<eos>'] = 0

    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        if (args.split == img['split']) or \
                (args.split == 'train' and img['split'] == 'restval') or \
                (args.split == 'all'):
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                tmp_tokens = sent['tokens'] + ['<eos>']
                tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='data/dataset_coco.json')
    parser.add_argument('--vocab_json', default='data/vocab.json')
    parser.add_argument('--output_pkl', default='data/coco-train.p')
    parser.add_argument('--split', default='train', help='test, val, train, all')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.input_json) as fid:
        imgs = json.load(fid)
    with open(args.vocab_json) as fid:
        itow = json.load(fid)
    wtoi = {w: i for i, w in itow.items()}
    imgs = imgs['images']
    _, ngram_idxs, ref_len = build_dict(imgs, wtoi, args)
    with open(args.output_pkl, 'w') as fid:
        pickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, fid, protocol=pickle.HIGHEST_PROTOCOL)
