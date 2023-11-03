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
from datasets.ondemandISIC import OnDemandISIC2017, for_train_transform, test_transform
import argparse
import warnings
from network.CoTrFuse import SwinUnet as Vit
import numpy as np 
from torchinfo import summary

#Plotting ------------------------- not in the original code
import matplotlib.pyplot as plt


#Clear the cache
torch.cuda.empty_cache()

#Parser
#Paths are set to the Google Drive paths (Francesco Di Gangi's Google Drive)

#Directories
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/training',
                    help='imgs train data path.')
parser.add_argument('--labels_train_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/training/gt',
                    help='labels train data path - ground truth.')
parser.add_argument('--csv_dir_train', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/training_tiny.csv',
                    help='labels train data path.')
parser.add_argument('--imgs_val_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/validation',
                    help='imgs val data path.')
parser.add_argument('--labels_val_path', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/validation/gt',
                    help='labels val data path - ground truth.')
parser.add_argument('--csv_dir_val', type=str,
                    default='/content/drive/MyDrive/cotrfuse_drive/validation_tiny.csv',
                    help='labels val data path.')

#Settings (batch size, workers, learning rate, epochs, num classes, yaml file, device)
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

#Other options (zip, cache, resume, accumulations chekpoint, optimization)
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

#Name of the tested model
'''
Please to understand which model you can use, refer to this github page
https://github.com/qubvel/segmentation_models.pytorch
'''
parser.add_argument('--model_name', type=str, default="efficientnet-b3", )

#Train starts
print("Starting preliminary training operations...")
args = parser.parse_args()
config = get_config(args)
begin_time = time.time()
set_seed(seed=2021)
device = args.device
epochs = args.warm_epoch + args.end_epoch

#CSV files for train and validation data
print("Getting labels and images path")
train_csv = args.csv_dir_train
df_val=args.csv_dir_val
#Variables that contains path to images and labels for both training and validation
train_imgs, train_masks = args.imgs_train_path, args.labels_train_path
val_imgs, val_masks = args.imgs_val_path, args.labels_val_path

'''
train_transform is a function that applies transformations to the training set,
test_transform is a function that applies transformations to the validation set.
best_acc_final is the best accuracy obtained from the training set and validation phase.
'''
train_transform = for_train_transform()
test_transform = test_transform
best_acc_final = []

#Training function 
def train(model, save_name):
    #preparing the dir where the model will be saved
    model_savedir = args.checkpoint + save_name + '/'
    save_name = model_savedir + 'ckpt'
    print("This is the folder where the model will be saved: ",model_savedir)
    #if does not exist, create it
    if not os.path.exists(model_savedir):
        os.mkdir(model_savedir)
    #Creating the dataset 'on demand' where the images are loaded only when needed
    train_ds=OnDemandISIC2017(train_csv,train_imgs, train_masks,train_transform)
    val_ds=OnDemandISIC2017(df_val, val_imgs, val_masks,test_transform,training=False)
    #due to heaviness of the model, sometimes we need to switch to cuda 
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().to('cuda')
    else:
        criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    #creating the dataloader object
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=False, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=False, num_workers=0)
    best_acc = 0
    #to plot for the accuracy ------------------------- not in the original code
    accuracies = []
    print("Training is about to start...")
    with tqdm(total=epochs, ncols=60) as t:
        for epoch in range(epochs):
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch, epochs, model, train_dl, val_dl, device, criterion, optimizer, CosineLR)
            f = open(model_savedir + 'log' + '.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss' + str(epoch_loss) + '  _val_loss' + str(epoch_val_loss) +
                    ' _epoch_acc' + str(epoch_iou) + ' _val_iou' + str(epoch_val_iou) + '\n')
            if epoch_val_iou > best_acc:
                f.write('\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name, '.pth']))
            accuracies.append(epoch_val_iou)
            f.close()
            t.update(1)
    #Plotting the graphs ------------------------- not in the original code
    plt.figure()
    plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    #write the file and close
    write_options(model_savedir, args, best_acc)
    print('Training over')
    #clear cache
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Main started in ISIC2017_segmentation_training.py")
    #if cuda is available, use cuda
    if torch.cuda.is_available():
        model = Vit(config, img_size=args.img_size, model_name=args.model_name, num_classes=args.num_classes).cuda()
    else:
        model = Vit(config, img_size=args.img_size, model_name=args.model_name, num_classes=args.num_classes)
    print("Model created (vit)")
    print("Charging config file")
    model.load_from(config)
    print("Summary about the model: \n")
    summary(model,input_size=(16,3,512,512))
    print("Charged config file")
    print("The encoder will be ",args.model_name)
    save_string="CoTrFuse_ISIC_"+args.model_name
    train(model, save_string)
    torch.cuda.empty_cache()
    print("Task completed.")