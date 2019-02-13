import sys
sys.path.append('coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
import numpy as np


def evaluate_diversity(tokenizer, capsById):
    n_caps_perimg = len(capsById[capsById.keys()[0]])
    capsById = tokenizer.tokenize(capsById)
    divs = []
    ngrams = []
    for i in range(1, 5):
        div = compute_div_n(capsById, i)
        divs.append(div)
        ngram = count_ngram(capsById, i)
        ngrams.append(ngram)
    num_words, average_length = count_words(capsById)
    mbleu = compute_mbleu(capsById, n_caps_perimg)
    print('num words: {}, average length: {}'.format(num_words, average_length))
    print('diversity: {}'.format(divs))
    print('num n-gram: {}'.format(ngrams))
    print('mbleu: {}'.format(mbleu))


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def compute_div_n(caps, n):
    aggr_div = []
    for k in caps:
        all_ngrams = set()
        lenT = 0.
        for c in caps[k]:
            tkns = c.split()
            lenT += len(tkns)
            ng = find_ngrams(tkns, n)
            all_ngrams.update(ng)
        aggr_div.append(float(len(all_ngrams)) / (1e-6 + float(lenT)))
    return np.array(aggr_div).mean()


def count_ngram(caps, n):
    all_ngrams = set()
    for k in caps:
        for c in caps[k]:
            tkns = c.split()
            ng = find_ngrams(tkns, n)
            all_ngrams.update(ng)
    return len(all_ngrams)


def count_words(caps):
    num_words = 0
    num_caps = 0
    for k in caps:
        num_caps += len(caps[k])
        for c in caps[k]:
            tkns = c.split()
            num_words += len(tkns)
    return num_words, float(num_words) / num_caps


def compute_mbleu(capsById, n_caps_perimg):
    scorer = Bleu(4)
    all_scrs = []
    for i in range(n_caps_perimg):
        tempRefsById = {}
        candsById = {}
        for k in capsById:
            tempRefsById[k] = capsById[k][:i] + capsById[k][i + 1:]
            candsById[k] = [capsById[k][i]]
        score, scores = scorer.compute_score(tempRefsById, candsById)
        all_scrs.append(score)
    all_scrs = np.array(all_scrs)
    mbleu = all_scrs.mean(axis=0)
    return mbleu
