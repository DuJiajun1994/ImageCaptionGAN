from torch.utils.data import Dataset
import numpy as np
import os


class COCOCaption(Dataset):
    def __init__(self, split, args):
        self.input_fc_dir = args.input_fc_dir
        self.input_att_dir = args.input_att_dir
        self.images = np.load('data/{}_images.npy'.format(split))

    def get_item(self, index):
        fc_feat = np.load(os.path.join(self.input_fc_dir, '{}.npy'.format(self.images[index])))
        att_feat = np.load(os.path.join(self.input_att_dir, '{}.npz'.format(self.images[index])))['feat']
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        item = {
            'fc_feats': fc_feat,
            'att_feats': att_feat,
            'att_masks': att_mask
        }
        return item

    def __len__(self):
        return len(self.images)


class TrainCaption(COCOCaption):
    def __init__(self, args):
        super(TrainCaption, self).__init__('train', args)
        self.labels = np.load('data/train_labels.npy')
        print('train: captions {}'.format(len(self.labels)))

    def __getitem__(self, index):
        item = self.get_item(index)
        label = self.labels[index]
        seq_len = (label > 0).sum() + 1
        mask = np.zeros(label.shape)
        mask[:seq_len] = 1
        item['labels'] = label
        item['masks'] = mask
        return item


class TestCaption(COCOCaption):
    def __init__(self, split, args):
        super(TestCaption, self).__init__(split, args)

    def __getitem__(self, index):
        item = self.get_item(index)
        return item
