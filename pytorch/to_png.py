import os
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

dir_data = 'water_segment'
train_img_path = os.path.join(dir_data, 'training/train_water_data_resized')
train_img_list = os.listdir(train_img_path)
train_img_list.sort()

train_label_path = os.path.join(dir_data, 'training/train_water_labeling_resized')
train_label_list = os.listdir(train_label_path)
train_label_list.sort()

for index in range(len(train_img_list)):
    img = io.imread(os.path.join(train_img_path, train_img_list[index]))
    label = io.imread(os.path.join(train_label_path, train_label_list[index]))
    img = img * 255
    label = label * 255
    io.imsave(f'./water_segment/training/train_water_data_resized_png/{train_img_list[index][:-4]}.png', img)
    io.imsave(f'./water_segment/training/train_water_labeling_resized_png/{train_label_list[index][:-4]}.png', label)

train_img_path = os.path.join(dir_data, 'validation/validate_water_data_resized')
train_img_list = os.listdir(train_img_path)
train_img_list.sort()

train_label_path = os.path.join(dir_data, 'validation/validate_water_labeling_resized')
train_label_list = os.listdir(train_label_path)
train_label_list.sort()

for index in range(len(train_img_list)):
    img = io.imread(os.path.join(train_img_path, train_img_list[index]))
    label = io.imread(os.path.join(train_label_path, train_label_list[index]))
    img = img * 255
    label = label * 255
    io.imsave(f'./water_segment/validation/validate_water_data_resized_png/{train_img_list[index][:-4]}.png', img)
    io.imsave(f'./water_segment/validation/validate_water_labeling_resized_png/{train_label_list[index][:-4]}.png',
              label)
