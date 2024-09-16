import os
import torch
import copy
import time
from tqdm import tqdm
from config import get_config
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from original_code.fit_COV import fit, set_seed, write_options
from datasets.dataset_COV import for_train_transform, test_transform, Mydataset
import argparse
import warnings
from network.CoTrFuse import SwinUnet as Vit
import matplotlib.pyplot as plt
from datetime import date

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', type=str,
                    default='',
                    help='imgs train data path.')
parser.add_argument('--labels_train_path', type=str,
                    default='',
                    help='labels train data path.')
parser.add_argument('--imgs_train_list', type=str, default='', help="Train csv")
parser.add_argument('--all_data_list', type=str,
                    default='', )
# -
parser.add_argument('--batch_size', default=16, type=int, help='batchsize')
parser.add_argument('--workers', default=16, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=50, type=int, )
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--resize', default=224, type=int, )
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=
'configs/swin_tiny_patch4_window7_224_lite_1.yaml')
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
parser.add_argument('--checkpoint', type=str, default='checkpoint/', )
#Name of the tested model
#----------------------------------------------
parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50','efficientnet-b3','efficientnet-b0'],
                    help='mixed precision opt level, if O0, no amp is used')
args = parser.parse_args()
config = get_config(args)

begin_time = time.time()
set_seed(seed=2021)
device = args.device
epochs = args.warm_epoch + args.end_epoch

train_csv = args.imgs_train_list
dataset_all = np.load(args.all_data_list) #dataset_all but only validation
imgs_val, masks_val = dataset_all['val_images'], dataset_all['val_labels']

df_train = pd.read_csv(train_csv)
all_train_imgs = df_train['image_name']
train_imgs, train_masks = args.imgs_train_path, args.labels_train_path
train_imgs = [''.join([train_imgs, '/', i]) for i in all_train_imgs]
train_masks = [''.join([train_masks, '/', i]) for i in all_train_imgs]
imgs_train = [np.load(i) for i in train_imgs]
masks_train = [np.load(i) for i in train_masks]

print('image done')

train_transform = for_train_transform()
test_transform = test_transform
best_acc_final = []


def train(model, save_name):
    model_savedir = args.checkpoint + save_name + '/'
    save_name = model_savedir + 'ckpt'
    print(model_savedir)
    if not os.path.exists(model_savedir):
        os.mkdir(model_savedir)

    train_ds = Mydataset(imgs_train, masks_train, train_transform)
    val_ds = Mydataset(imgs_val, masks_val, test_transform)

    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=False, num_workers=8,
                          drop_last=True, )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=False, num_workers=8, )
    #to plot for the accuracy ------------------------- not in the original code
    accuracies = []
    train_losses = []
    val_losses = []
    epoch_accuracies=[]
    best_acc = 0
    # ------------------------------------------------------------------------------
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
            train_losses.append(epoch_loss)
            val_losses.append(epoch_val_loss)
            epoch_accuracies.append(epoch_iou)
            f.close()
            t.update(1)
    #Plotting the graphs ------------------------- not in the original code
    # ----------------------------------------------------------------------------------    
    # IoU
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-')
    plt.title('Accuracy over Epochs '+ args.model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(save_name+"_iou.png")
    plt.show()
    # losses
    plt.figure(figsize=(8, 4))
    #plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-')
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss '+args.model_name)
    plt.savefig(save_name+"_losses.png")
    plt.show()
    # epoch accuracies
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), epoch_accuracies, label='Epoch Accuracy ', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch Accuracy - Epochs '+ args.model_name)
    plt.savefig(save_name+"_epoch_accuracies.png")
    plt.show()
    # Calculate average values
    avg_epoch_loss = sum(train_losses) / len(train_losses)
    avg_epoch_iou = sum(epoch_accuracies) / len(epoch_accuracies)
    avg_epoch_val_loss = sum(val_losses) / len(val_losses)
    avg_epoch_val_iou = sum(accuracies) / len(accuracies)
    print("Average Train Loss:", avg_epoch_loss)
    print("Average Train IoU:", avg_epoch_iou)
    print("Average Validation Loss:", avg_epoch_val_loss)
    print("Average Validation IoU:", avg_epoch_val_iou)
    # ---- write on file ----
    with open("averages_cov.txt", "w") as f:
        f.write(f"Average Train Loss: {avg_epoch_loss}\n")
        f.write(f"Average Train IoU: {avg_epoch_iou}\n")
        f.write(f"Average Validation Loss: {avg_epoch_val_loss}\n")
        f.write(f"Average Validation IoU: {avg_epoch_val_iou}\n")
    print(f"Metrics saved to avergaes_cov.txt")
    # ----------------------------------------------------------------------------------
    #write the file and close
    write_options(model_savedir, args, best_acc)
    print('Training over')
    #clear cache
    torch.cuda.empty_cache()


if __name__ == '__main__':
    model = Vit(config, model_name=args.model_name, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_from(config)
    model.summary()
    today=date.today()
    str_today=str(today)
    str_model_name=str(args.model_name)
    save_string="CoTrFuse_COV_"+str_today+"_"+str_model_name
    train(model, 'CoTrFuse/COV')
