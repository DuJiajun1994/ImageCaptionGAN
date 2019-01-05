from torch.utils.data import Dataset
import numpy as np
import os
import random


class ImageDataset(Dataset):
    def __init__(self, split, expand_by_labels, args):
        self.max_att_size = 100
        self.input_fc_dir = args.input_fc_dir
        self.input_att_dir = args.input_att_dir
        if expand_by_labels:
            self.images = np.load('data/{}_images.npy'.format(split))
        else:
            self.images = np.load('data/raw_{}_images.npy'.format(split))

    def __getitem__(self, index):
        fc_feat = np.load(os.path.join(self.input_fc_dir, '{}.npy'.format(self.images[index]))).astype(np.float32)
        origin_att_feat = np.load(os.path.join(self.input_att_dir, '{}.npz'.format(self.images[index])))['feat']
        origin_att_feat = origin_att_feat.reshape(-1, origin_att_feat.shape[-1])
        att_size = origin_att_feat.shape[0]
        att_feat = np.zeros([self.max_att_size, origin_att_feat.shape[-1]], dtype=np.float32)
        att_mask = np.zeros([self.max_att_size], dtype=np.float32)
        att_feat[:att_size] = origin_att_feat
        att_mask[:att_size] = 1
        item = {
            'images': self.images[index],
            'fc_feats': fc_feat,
            'att_feats': att_feat,
            'att_masks': att_mask
        }
        return item

    def __len__(self):
        return len(self.images)


class CaptionDataset(ImageDataset):
    def __init__(self, split, args):
        super(CaptionDataset, self).__init__(split=split, expand_by_labels=True, args=args)
        self.labels = np.load('data/{}_labels.npy'.format(split))

    def __getitem__(self, index):
        item = super(CaptionDataset, self).__getitem__(index)
        item['labels'] = self.labels[index]
        return item


class DiscCaption(CaptionDataset):
    def __init__(self, split, args):
        super(DiscCaption, self).__init__(split, args)

    def __getitem__(self, index):
        item = super(DiscCaption, self).__getitem__(index)
        num_labels = len(self.labels)
        wrong_index = random.randint(0, num_labels - 1)
        while self.images[wrong_index] == self.images[index]:
            wrong_index = random.randint(0, num_labels - 1)
        item['wrong_labels'] = self.labels[wrong_index]
        item['match_labels'] = self._get_match_label(index)
        return item

    def _get_match_label(self, index):
        mi = max(index - 5, 0)
        ma = min(index + 5, len(self.labels) - 1)
        match_index = random.randint(mi, ma)
        while match_index == index or self.images[match_index] != self.images[index]:
            match_index = random.randint(mi, ma)
        return self.labels[match_index]


class GeneratorCaption(ImageDataset):
    def __init__(self, split, args):
        super(GeneratorCaption, self).__init__(split=split, expand_by_labels=False, args=args)
        self.labels = np.load('data/raw_{}_labels.npy'.format(split))

    def __getitem__(self, index):
        item = super(GeneratorCaption, self).__getitem__(index)
        item['labels'] = self.labels[index]
        return item
