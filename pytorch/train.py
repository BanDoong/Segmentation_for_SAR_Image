# coding: utf8

from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from read_data import *
from model import unet, att_unet, nested_unet, nested_resunet, ResUnet, colonSegNet, DeepLab, HRNet, DDANet
from tqdm import tqdm
from time import time
from torchvision import transforms


# from apex.parallel import DistributedDataParallel as DDP
# MULTI GPU
# if (device.type =='cuda') and (torch.cuda.device_count()>1):
#     print("Multi GPU ACTIVATES")
#     net = nn.DataParallel(net)
#     classifier = nn.DataParallel(classifier)
# net.to(device)
# classifier.to(device)

class Train:
    def __init__(self, args):
        self.model = args.model
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_worker = args.num_worker
        self.weight_decay = args.weight_decay
        self.dir_data = args.dir_data
        self.dir_ckpt = args.dir_ckpt
        self.dir_log = args.dir_log
        self.gpus = args.gpus
        self.dir_out = args.dir_out

        self.device = torch.device(f'cuda:{self.gpus}' if torch.cuda.is_available() else 'cpu')
        # torch.distributed.init_process_group(backend='nccl')

    def train(self):
        lr = self.lr
        weight_decay = self.weight_decay
        batch_size = self.batch_size
        device = self.device
        num_epoch = self.num_epoch
        num_worker = self.num_worker
        dir_data = self.dir_data
        model = self.model
        dir_log = self.dir_log
        dir_out = self.dir_out
        dir_ckpt = self.dir_ckpt
        fn_loss_MSE = nn.MSELoss().to(device)

        ## 네트워크 생성
        # CPU GPU 결정 위해 to Device 사용

        if model == 'unet':
            net = unet()
        elif model == 'att_unet':
            net = att_unet()
        elif model == 'nested_unet':
            net = nested_unet()
        elif model == 'ResUnet':
            net = ResUnet()
        elif model == 'nested_resunet':
            net = nested_resunet()
        elif model == 'colonSegNet':
            net = colonSegNet()
        elif model == 'DeepLab':
            net = DeepLab()
        elif model == 'HRNet':
            net = HRNet()
        elif model == 'DDANet':
            net = DDANet()

        net.to(device)
        params_to_mri = [{'params': net.parameters()}]
        optim = torch.optim.Adam(params_to_mri, lr=lr, weight_decay=weight_decay)

        st_epoch = 0

        transform = transforms.Compose([Normalization(normal=True), ToTensor()])
        dataset_train = Dataset(dir_data=dir_data, transform=transform, dataset='training')
        dataset_val = Dataset(dir_data=dir_data, transform=transform, dataset='validation')
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                  pin_memory=True, drop_last=True)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                pin_memory=True, drop_last=True)
        loss_arr = []
        loss_arr_val = []

        for epoch in range(st_epoch + 1, num_epoch + 1):
            net.train()
            for batch_idx, data in enumerate(tqdm(loader_train), start=1):
                img = data['img'].to(device)
                label = data['label'].to(device)

                output = net(img)

                loss = fn_loss_MSE(output, label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_arr.append(loss.item())

            train_loss = np.mean(loss_arr)

            net.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(loader_val)):
                    img = data['img'].to(device)
                    label = data['label'].to(device)

                    output = net(img)

                    loss_val = fn_loss_MSE(output, label)
                    loss_arr_val.append(loss_val.item())

            val_loss = np.mean(loss_arr_val)

            output_list = [epoch, train_loss, val_loss]

            # output_list = [epoch, train_loss, val_loss, dsc, mIOU, recall, precision]

            print(output_list)

        save_path = f'{dir_out}/net-{str(net)}.pth'
        torch.save(net.state_dict(), save_path)

        # writer_train.close()
        # writer_val.close()
