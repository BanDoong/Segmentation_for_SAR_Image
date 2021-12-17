import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool, img_as_ubyte
from tqdm import tqdm
import matplotlib.pyplot as plt

train_path = './water_segment/training/train_water_data'
label_path = './water_segment/training/train_water_labeling'
test_path = './water_segment/validation/validate_water_data'
test_label_path = './water_segment/validation/validate_water_labeling'


def resize_img(path):
    data_list = os.listdir(path)
    for name in tqdm(data_list):
        total_path = os.path.join(path, name)
        img = io.imread(total_path)
        # img = img_as_bool(resize(img, (256, 256)))
        img = resize(img, (256, 256))
        io.imsave(path + '_resized/' + name, img_as_ubyte(img), check_contrast=False)


# resize_img(train_path)
resize_img(label_path)
# resize_img(test_path)
resize_img(test_label_path)

