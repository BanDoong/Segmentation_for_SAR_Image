from modules import *
import torch.nn as nn
from torchinfo import summary
import os
import matplotlib.pyplot as plt
import numpy as np

dir_data = './water_segment'


train_img_path = os.path.join(dir_data, 'training/train_water_data')
train_img_list = os.listdir(train_img_path)
img = plt.imread(os.path.join(train_img_path, train_img_list[0]))
print(img.shape)
img = np.repeat(img[np.newaxis, :, :], 3, 0)
print(img.shape)