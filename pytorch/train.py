# coding: utf8
import numpy as np
import timm
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from read_data import *
from model import unet, att_unet, nested_unet, ResUnet, HRNet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm
from time import time
from torchvision import transforms
# from metric import SegmentationMetric, dice_loss
from torchvision.utils import save_image
from torchvision.transforms import Normalize
from losses import IoULoss, DiceLoss, DiceBCELoss
from torchvision import models
from losses import dice_coef_torch, iou_seg_torch, pix_acc_torch
from main import fix_seed

fix_seed()

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
        self.pretrain = args.pretrain
        self.patience = args.patience
        self.aug = args.aug
        self.ls = args.ls

        self.device = torch.device(f'cuda:{self.gpus}' if torch.cuda.is_available() else 'cpu')
        self.writer_train = SummaryWriter(f'./{self.dir_ckpt}/{self.model}')


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

        # fn_loss_MSE = nn.MSELoss().to(device)
        # fn_loss_BCE = nn.BCEWithLogitsLoss().to(device)
        # dice = DiceLoss().to(device)
        # iou = IoULoss().to(device)

        if self.ls =='iou':
            criterian = IoULoss().to(device)
        else:
            criterian = DiceBCELoss().to(device)
        fn_class = lambda x: 1.0 * (x > 0.5)

        def to_cpu(data):
            return data.cpu().clone().detach().numpy()
        ## 네트워크 생성
        # CPU GPU 결정 위해 to Device 사용

        if model == 'unet':
            net = unet(in_ch=3, out_ch=1)
        elif model == 'att_unet':
            net = att_unet(img_ch=3)
        elif model == 'nested_unet':
            net = nested_unet()
        elif model == 'resunet':
            net = ResUnet(channel=3)
        elif model == 'deeplab':
            net = models.segmentation.deeplabv3_resnet101(pretrained=False)
            net.classifier = DeepLabHead(2048, 1)
            net.classifier[4].out_channels = 1

        elif model == 'HRNet':
            # net = HRNet()
            if self.pretrain:
                net = timm.create_model('hrnet_w48', pretrained=False, num_classes=256 * 256)
            else:
                net = timm.create_model('hrnet_w48', pretrained=True, num_classes=256 * 256)

        def save(path, net=None, optim=None, pretrain=None):
            if pretrain:
                torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, path)
            else:
                torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, path)

        # MULTI GPU

            # net = nn.DataParallel(net)

        net.to(device)
        params_to_mri = [{'params': net.parameters()}]
        optim = torch.optim.Adam(params_to_mri, lr=lr, weight_decay=weight_decay)

        st_epoch = 0

        transform = transforms.Compose([RandomFlip(), ToTensor()])
        # transform = transforms.Compose([ToTensor()])
        dataset_train = Dataset(dir_data=dir_data, transform=transform, dataset='training', model=self.model,
                                aug=self.aug)
        dataset_val = Dataset(dir_data=dir_data, transform=transform, dataset='validation', model=self.model,
                              aug=self.aug)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                  pin_memory=True, drop_last=True)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                pin_memory=True, drop_last=True)
        loss_arr = []
        loss_arr_val = []
        from utils import EarlyStopping
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            train_dice = []
            val_dice = []
            train_iou = []
            val_iou = []
            train_acc = []
            val_acc = []
            net.train()
            for batch_idx, data in enumerate(tqdm(loader_train), start=1):
                img = data['img'].to(device)
                label = data['label'].to(device)
                output = net(img)
                if self.model == 'deeplab':
                    loss = criterian(output['out'], label)
                    output = fn_class(output['out'])
                elif self.model == 'HRNet':
                    output = torch.reshape(output, (batch_size, 1, 256, 256))
                    loss = criterian(output, label)
                    output = fn_class(output)
                else:
                    loss = criterian(output, label)
                    output = fn_class(output)

                train_dice.append(to_cpu(dice_coef_torch(label, output)))
                train_iou.append(to_cpu(iou_seg_torch(label, output)))
                train_acc.append(to_cpu(pix_acc_torch(label, output)))
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_arr.append(loss.item())

                if epoch % 10 == 0 and batch_idx % 10 == 0:
                    input_tensor = img[0]
                    output_tensor = fn_class(output[0])
                    label_tensor = label[0]
                    save_image(input_tensor,
                               f'./{self.dir_ckpt}/{self.model}/train/input/batch_idx_{str(batch_idx)}_epoch_{str(epoch)}.png')
                    save_image(output_tensor,
                               f'./{self.dir_ckpt}/{self.model}/train/output/batch_idx_{str(batch_idx)}_epoch_{str(epoch)}.png')
                    save_image(label_tensor,
                               f'./{self.dir_ckpt}/{self.model}/train/label/batch_idx_{str(batch_idx)}_epoch_{str(epoch)}.png')
                    self.writer_train.add_image('image/train_output', output, epoch, dataformats='NCHW')
                    self.writer_train.add_image('image/train_label', label, epoch, dataformats='NCHW')

            train_loss = np.mean(loss_arr)
            train_dice = np.mean(train_dice)
            train_iou = np.mean(train_iou)
            train_acc = np.mean(train_acc)
            self.writer_train.add_scalar('loss/train_Dice_CE_loss', train_loss, epoch)
            self.writer_train.add_scalar('loss/train_Dice', train_dice, epoch)
            self.writer_train.add_scalar('loss/train_iou', train_iou, epoch)
            self.writer_train.add_scalar('loss/train_acc', train_acc, epoch)

            net.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(loader_val)):
                    img = data['img'].to(device)
                    label = data['label'].to(device)

                    output = net(img)

                    if self.model == 'deeplab':
                        loss_val = criterian(output['out'], label)
                        output = fn_class(output['out'])
                    elif self.model == 'HRNet':
                        output = torch.reshape(output, (batch_size, 1, 256, 256))
                        loss_val = criterian(output, label)
                        output = fn_class(output)
                    else:
                        loss_val = criterian(output, label)
                        output = fn_class(output)

                    val_dice.append(to_cpu(dice_coef_torch(label, output)))
                    val_iou.append(to_cpu(iou_seg_torch(label, output)))
                    val_acc.append(to_cpu(pix_acc_torch(label,output)))
                    loss_arr_val.append(loss_val.item())

                    if epoch % 20 == 0 and batch_idx % 10 == 0:
                        input_tensor = img[0]
                        output_tensor = fn_class(output[0])
                        label_tensor = label[0]
                        save_image(input_tensor,
                                   f'./{self.dir_ckpt}/{self.model}/val/input/batch_idx_{str(batch_idx)}_epoch_{str(epoch)}.png')
                        save_image(output_tensor,
                                   f'./{self.dir_ckpt}/{self.model}/val/output/batch_idx_{str(batch_idx)}_epoch_{str(epoch)}.png')
                        save_image(label_tensor,
                                   f'./{self.dir_ckpt}/{self.model}/val/label/batch_idx_{str(batch_idx)}_epoch_{str(epoch)}.png')
                        self.writer_train.add_image('image/val_output', output, epoch, dataformats='NCHW')
                        self.writer_train.add_image('image/val_label', label, epoch, dataformats='NCHW')

            val_loss = np.mean(loss_arr_val)
            val_dice = np.mean(val_dice)
            val_iou = np.mean(val_iou)
            val_acc = np.mean(val_acc)
            self.writer_train.add_scalar('loss/val_Dice_CE_loss', val_loss, epoch)
            self.writer_train.add_scalar('loss/val_Dice', val_dice, epoch)
            self.writer_train.add_scalar('loss/val_iou', val_iou, epoch)
            self.writer_train.add_scalar('loss/val_acc', val_acc, epoch)

            output_list = [epoch, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, train_acc, val_acc]

            print(output_list)
            early_stopping(val_loss, net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        save_path = f'{dir_out}/net-{self.model}.pth'
        save(save_path, net=net, optim=optim, pretrain=False)

        self.writer_train.close()
