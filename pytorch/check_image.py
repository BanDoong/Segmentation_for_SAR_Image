import os
import matplotlib.pyplot as plt

train_path = 'ckpt/att_unet/train'
val_path = 'ckpt/att_unet/val'
data_type = ['input', 'label', 'output']

row = 3
col = 5
i = 1
j = 6
k = 11
ax = []
# constrained_layout=True
fig = plt.figure(figsize=(12, 12),constrained_layout=True)

for epoch in range(20, 120, 20):
    input_img = plt.imread(os.path.join(train_path, data_type[0], f'batch_idx_20_epoch_{str(epoch)}.png'))
    label_img = plt.imread(os.path.join(train_path, data_type[1], f'batch_idx_20_epoch_{str(epoch)}.png'))
    output_img = plt.imread(os.path.join(train_path, data_type[2], f'batch_idx_20_epoch_{str(epoch)}.png'))

    ax1 = fig.add_subplot(row, col, i)
    ax1.imshow(input_img, cmap='gray')

    ax2 = fig.add_subplot(row, col, j)
    ax2.imshow(label_img, cmap='gray')

    ax3 = fig.add_subplot(row, col, k)
    ax3.imshow(output_img, cmap='gray')
    ax1.set_title(f'Epoch : {str(epoch)}')
    if i == 1:
        ax1.set_ylabel('input')
        ax2.set_ylabel('Label')
        ax3.set_ylabel('Output')
    i += 1
    j += 1
    k += 1
plt.show()
