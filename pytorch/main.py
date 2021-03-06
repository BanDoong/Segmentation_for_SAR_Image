import argparse
from train import *
from read_data import *
from utils import *
import random
import torch.backends.cudnn as cudnn

"""
python  main.py \
        --mode train \
        --modality mri tau \
        --dir_label getlabels_AD_CN \
        --num_label 2 \ 
        --num_workers 2 \
        --batch_size 4
        --epochs 100 \
        --lr 1e-4 \
        --group classification \
        --dir_data ./caps/subjects \
        --dir_ckpt ./ckpt \
"""


## Reproducibility

# # torch random seed 고정 ==> reproducible
# torch.manual_seed(1)
# torch.cuda.manual_seed(2)
# torch.cuda.manual_seed_all(3) # if use multi-GPU
# # cudnn randomness 고정
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # numpy random 고정
# np.random.seed(4)
# # torchvision random 고정
# random.seed(5)

def fix_seed():
    # torch random seed 고정 ==> reproducible
    torch.manual_seed(1)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(3)  # if use multi-GPU
    # cudnn randomness 고정
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # numpy random 고정
    np.random.seed(4)
    # torchvision random 고정
    random.seed(50)


parser = argparse.ArgumentParser(description='Train the my Network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', default='unet',
                    choices=['unet', 'nested_unet', 'att_unet', 'resunet', 'deeplab', 'HRNet'],
                    dest='model')

parser.add_argument('--num_epoch', type=int, default=100, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=32, dest='batch_size')
parser.add_argument('--lr', default=1e-4, dest='lr', type=float, help='learning rate')
parser.add_argument('--num_worker', type=int, default=12, dest='num_worker')
parser.add_argument('--weight_decay', default=1e-5, dest='weight_decay')

parser.add_argument('--dir_data', default='./water_segment', dest='dir_data')
parser.add_argument('--dir_ckpt', default='./ckpt', dest='dir_ckpt')
parser.add_argument('--dir_log', default='./logs', dest='dir_log')
parser.add_argument('--gpus', default=0, type=int, dest='gpus')
parser.add_argument('--patience', default=100, type=int, dest='patience')
# output directory
parser.add_argument('--dir_out', default='./output', dest='dir_out')
parser.add_argument('--ls', default='dice', choices=['dice', 'iou'], dest='ls')
parser.add_argument('--pretrain', action='store_true', dest='pretrain')
parser.add_argument('--aug', action='store_true', dest='aug')

PARSER = Parser(parser)


def main():
    ARGS = PARSER.get_arguments()

    TRAINER = Train(ARGS)

    TRAINER.train()


if __name__ == '__main__':
    main()
