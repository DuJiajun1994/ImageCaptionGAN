from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle
import argparse
import numpy as np


def build_vocab(imgs, count_thr):
    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def save_train_captions(imgs, wtoi, max_length):
    images = []
    labels = []
    references = {}
    cnt = 0
    for img in imgs:
        image_id = img['cocoid']
        if img['split'] == 'train' or img['split'] == 'restval':
            cnt += 1
            reference = []
            for caption in img['final_captions']:
                label = np.zeros(max_length, dtype=np.int)
                for i, w in enumerate(caption):
                    if i < max_length:
                        label[i] = wtoi[w]
                images.append(image_id)
                labels.append(label)
                reference.append(get_reference(label))
            references[image_id] = reference
    images = np.array(images)
    labels = np.array(labels)
    print('train: images {}, captions {}'.format(cnt, images.shape[0]))
    np.save('data/train_images.npy', images)
    np.save('data/train_labels.npy', labels)
    with open('data/train_references.pkl', 'w') as fid:
        pickle.dump(references, fid)


def get_reference(seq):
    words = []
    for word in seq:
        words.append(str(word))
        if word == 0:
            break
    caption = ' '.join(words)
    return caption


def save_images(imgs, split):
    images = []
    for img in imgs:
        if img['split'] == split:
            images.append(img['cocoid'])
    images = np.array(images)
    print('{}: images {}'.format(split, images.shape[0]))
    np.save('data/{}_images.npy'.format(split), images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='data/dataset_coco.json')
    parser.add_argument('--max_length', default=16, type=int)
    parser.add_argument('--word_count_threshold', default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    imgs = json.load(open(args.input_json, 'r'))
    imgs = imgs['images']

    # create the vocab
    vocab = build_vocab(imgs, args.word_count_threshold)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    save_train_captions(imgs, wtoi, args.max_length)
    save_images(imgs, 'val')
    save_images(imgs, 'test')
    with open('data/vocab.json', 'w') as fid:
        json.dump(itow, fid)
