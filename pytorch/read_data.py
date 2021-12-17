import os
import torch
import numpy as np
from skimage import io
from skimage.color import gray2rgb
from main import fix_seed

fix_seed()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, model, dataset, transform=None, aug=False):
        self.dataset = dataset
        self.dir_data = dir_data
        self.transform = transform
        self.model = model
        self.aug = aug

        if aug:
            self.train_img_path = os.path.join(dir_data, 'training/train_aug')
            self.train_label_path = os.path.join(dir_data, 'training/train_label_aug')
        else:
            self.train_img_path = os.path.join(dir_data, 'training/train_water_data_resized_png')
            self.train_label_path = os.path.join(dir_data, 'training/train_water_labeling_resized_png')

        self.train_img_list = os.listdir(self.train_img_path)
        self.train_img_list.sort()

        self.train_label_list = os.listdir(self.train_label_path)
        self.train_label_list.sort()

        self.validation_img_path = os.path.join(dir_data, 'validation/validate_water_data_resized_png')
        self.validation_img_list = os.listdir(self.validation_img_path)
        self.validation_img_list.sort()

        self.validation_label_path = os.path.join(dir_data, 'validation/validate_water_labeling_resized_png')
        self.validation_label_list = os.listdir(self.validation_label_path)
        self.validation_label_list.sort()

    def __getitem__(self, index):
        # Read image & sort
        if self.dataset == 'training':
            img = io.imread(os.path.join(self.train_img_path, self.train_img_list[index]))
            img = gray2rgb(image=img)
            label = io.imread(os.path.join(self.train_label_path, self.train_label_list[index]))
        elif self.dataset == 'validation':
            img = io.imread(os.path.join(self.validation_img_path, self.validation_img_list[index]))
            img = gray2rgb(image=img)
            label = io.imread(os.path.join(self.validation_label_path, self.validation_label_list[index]))

        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        img = img / 255.0
        label = label / 255.0

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


class RandomFlip(object):
    def __call__(self, data):
        label, img = data['label'], data['img']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            img = np.fliplr(img)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            img = np.flipud(img)

        data = {'img': img, 'label': label}

        return data


class ToTensor(object):
    def __call__(self, data):
        label, img = data['label'], data['img']
        # numpy (x,y,channel) | pytorch (channel,x,y)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'img': torch.from_numpy(img), 'label': torch.from_numpy(label), }

        return data
