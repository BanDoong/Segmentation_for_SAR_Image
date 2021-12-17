import os

import numpy as np
from skimage import io
import elasticdeform
import matplotlib.pyplot as plt

np.random.seed(8)

orig_train_data_path = './water_segment/training/train_water_data_resized'
orig_train_label_path = './water_segment/training/train_water_labeling_resized'
orig_val_data_path = './water_segment/validation/validate_water_data_resized'
orig_val_label_path = './water_segment/validation/validate_water_labeling_resized'
path_list = [orig_train_data_path, orig_train_label_path, orig_val_data_path, orig_val_label_path]

train_data_list = os.listdir(orig_train_data_path)
train_label_list = os.listdir(orig_train_label_path)
val_data_list = os.listdir(orig_val_data_path)
val_label_list = os.listdir(orig_val_label_path)

path = './water_segment/training/train_water_labeling_resized'
path_2 = './water_segment/training/train_water_data_resized'
img = io.imread(os.path.join(path_2, 'WTR03330_K5_NIA0557.tif'))
label = io.imread(os.path.join(path, 'WTR03330_K5_NIA0557_label.tif')) * 255

img_out, label_out = elasticdeform.deform_random_grid([img, label], points=2, order=[3, 1])
img_out = img_out / 255
label_out = label_out / 255
io.imsave(orig_train_data_path + '_aug/WTR03330_K5_NIA0557_1.png', img_out)
io.imsave(orig_train_label_path + '_aug/WTR03330_K5_NIA0557_label_1.png', label_out)

path = './water_segment/training/train_water_labeling_resized'
path_2 = './water_segment/training/train_water_data_resized'
input = io.imread(os.path.join(path_2, 'WTR03330_K5_NIA0557.tif'))
label = io.imread(os.path.join(path, 'WTR03330_K5_NIA0557_label.tif'))

input_2 = io.imread(os.path.join(path_2 + '_aug', 'WTR03330_K5_NIA0557_1.png'))
label_2 = io.imread(os.path.join(path + '_aug', 'WTR03330_K5_NIA0557_label_1.png'))

row = 1
col = 4
fig = plt.figure()
ax1 = fig.add_subplot(row, col, 1)
ax2 = fig.add_subplot(row, col, 2)
ax3 = fig.add_subplot(row, col, 3)
ax4 = fig.add_subplot(row, col, 4)
ax1.imshow(input, cmap='gray')
ax1.set_title('Input')
ax2.imshow(label, cmap='gray')
ax2.set_title('Label')
ax3.imshow(input_2, cmap='gray')
ax3.set_title('input_2')
ax4.imshow(label_2, cmap='gray')
ax4.set_title('Label_2')
plt.show()

for i in range(len(train_data_list)):
    img = io.imread(os.path.join(orig_train_data_path, train_data_list[i]))
    label = io.imread(os.path.join(orig_train_label_path, train_label_list[i]))
    X_deformed, Y_deformed = elasticdeform.deform_random_grid([img, label], order=[3, 1])
    io.imsave(orig_train_data_path + '_aug/' + train_data_list[i][:-4] + '_1.png', X_deformed)
    io.imsave(orig_train_label_path + '_aug/' + train_label_list[i][:-4] + '_1.png', Y_deformed)

for i in range(len(val_data_list)):
    img = io.imread(os.path.join(orig_val_data_path, val_data_list[i]))
    label = io.imread(os.path.join(orig_val_label_path, val_label_list[i]))
    X_deformed, Y_deformed = elasticdeform.deform_random_grid([img, label], order=[3, 1])

    io.imsave(orig_val_data_path + '_aug/' + val_data_list[i][:-4] + '_1.png', X_deformed)
    io.imsave(orig_val_label_path + '_aug/' + val_label_list[i][:-4] + '_1.png', Y_deformed)
