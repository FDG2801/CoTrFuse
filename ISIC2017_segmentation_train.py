'''
The goal is to implement and tweak the code from CoTrFuse: a novel framework by fusing CNN and transformer for medical image segmentation,
the used dataset is ISIC2017, which is a dataset of skin cancer images. In this particular task,  we are going to segment the images.
The training set is of 2000 images, and the validation set is of 150 images. The images are of size 512x512. 
The code is implemented in Google Colab, and the dataset is stored in Google Drive. 

The training set dimension might variate, so the validation set dimension might variate as well. It depends on the machine used to run the code.
'''
#Libraries import
import cv2
import os
import torch
import copy
import time
from tqdm import tqdm
from config import get_config
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from fit_ISIC import fit, set_seed, write_options
#from datasets.dataset_ISIC import Mydataset, for_train_transform, test_transform
#from datasets.new_dataset_ISIC import ISIC2017, for_train_transform, test_transform
from datasets.ondemandISIC import OnDemandISIC2017, for_train_transform, test_transform
import argparse
import warnings
from network.CoTrFuse import SwinUnet as Vit
import numpy as np 
from torchinfo import summary

#Clear the cache
torch.cuda.empty_cache()

#Parser
#Paths are set to the Google Drive paths (Francesco Di Gangi's Google Drive)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/training',
                    help='imgs train data path.')
parser.add_argument('--labels_train_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/training/gt',
                    help='labels train data path.')
parser.add_argument('--csv_dir_train', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/training_tiny.csv',
                    help='labels train data path.')
parser.add_argument('--imgs_val_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/validation',
                    help='imgs val data path.')
parser.add_argument('--labels_val_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/validation/gt',
                    help='labels val data path.')
parser.add_argument('--csv_dir_val', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/validation_tiny.csv',
                    help='labels val data path.')
parser.add_argument('--batch_size', default=8, type=int, help='batchsize') #BATCH SIZE
parser.add_argument('--workers', default=16, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=3, type=int, )
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=
'/content/CoTrFuse/configs/swin_tiny_patch4_window7_224_lite.yaml')
parser.add_argument('--num_classes', '-t', default=2, type=int, )
parser.add_argument('--device', default='cuda', type=str, )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--checkpoint', type=str, default='/content/CoTrFuse/checkpoint/', )