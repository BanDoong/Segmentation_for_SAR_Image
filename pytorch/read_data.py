import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, dataset, transform=None):
        self.dataset = dataset
        self.dir_data = dir_data
        self.transform = transform

        self.train_img_path = os.path.join(dir_data, 'training/train_water_data')
        self.train_img_list = os.listdir(self.train_img_path)
        self.train_label_path = os.path.join(dir_data, 'training/train_water_labeling')
        self.train_label_list = os.listdir(self.train_label_path)

        self.validation_img_path = os.path.join(dir_data, 'validation/validate_water_data')
        self.validation_img_list = os.listdir(self.validation_img_path)
        self.validation_label_path = os.path.join(dir_data, 'validation/validate_water_labeling')
        self.validation_label_list = os.listdir(self.validation_label_path)

    def __getitem__(self, index):
        # Read image & sort
        if self.dataset == 'training':
            img = plt.imread(os.path.join(self.train_img_path, self.train_img_list[index]))
            img = np.repeat(img[np.newaxis, :, :], 3, 0)
            img = img.astype(np.float32)

            label = plt.imread(os.path.join(self.train_label_path, self.train_label_list[index]))
            label = np.repeat(label[np.newaxis, :, :], 3, 0)
            label = label.astype(np.float32)

        elif self.dataset == 'validation':
            img = plt.imread(os.path.join(self.validation_img_path, self.validation_img_list[index]))
            img = np.repeat(img[np.newaxis, :, :], 3, 0)
            img = img.astype(np.float32)

            label = Image.open(os.path.join(self.validation_label_path, self.validation_label_list[index]),
                               cv2.IMREAD_GRAYSCALE)
            label = np.repeat(label[np.newaxis, :, :], 3, 0)
            label = label.astype(np.float32)

        sample = {'img': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        if self.dataset == 'training':
            return len(self.train_img_list)
        else:
            return len(self.validation_img_list)

    def num_elem_per_image(self):
        return 2


class Normalization(object):
    def __init__(self, normal=True):
        self.normal = normal

    def __call__(self, data):
        img, label = data['img'], data['label']

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        data = {'img': img, 'label': label}

        return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        img, label = data['img'], data['label']

        img = img.transpose((3, 0, 1, 2)).astype(np.float32)
        label = label.transpose((3, 0, 1, 2)).astype(np.float32)

        data = {'img': img, 'label': label}

        return data
