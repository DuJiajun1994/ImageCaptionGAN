from torch.utils.data import Dataset
import numpy as np
import os


class ImageDataset(Dataset):
    def __init__(self, split, args):
        self.max_att_size = 100
        self.input_fc_dir = args.input_fc_dir
        self.input_att_dir = args.input_att_dir
        self.images = np.load('data/{}_images.npy'.format(split))

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


class TrainCaption(ImageDataset):
    def __init__(self, args):
        super(TrainCaption, self).__init__('train', args)
        self.labels = np.load('data/train_labels.npy')

    def __getitem__(self, index):
        item = super(TrainCaption, self).__getitem__(index)
        item['labels'] = self.labels[index]
        return item
