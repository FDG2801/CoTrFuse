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
from datasets.dataset_ISIC import Mydataset, for_train_transform, test_transform
import argparse
import warnings
from network.CoTrFuse import SwinUnet as Vit


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', type=str,
                    default='datasets_tiny/training',
                    help='imgs train data path.')
parser.add_argument('--labels_train_path', type=str,
                    default='datasets_tiny/training/gt',
                    help='labels train data path.')
parser.add_argument('--csv_dir_train', type=str,
                    default='training_tiny.csv',
                    help='labels train data path.')
parser.add_argument('--imgs_val_path', type=str,
                    default='datasets_tiny/validation',
                    help='imgs val data path.')
parser.add_argument('--labels_val_path', type=str,
                    default='datasets_tiny/validation/gt',
                    help='labels val data path.')
parser.add_argument('--csv_dir_val', type=str,
                    default='validation_tiny.csv',
                    help='labels val data path.')
parser.add_argument('--batch_size', default=8, type=int, help='batchsize')
parser.add_argument('--workers', default=16, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=5, type=int, )
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=
'configs/swin_tiny_patch4_window7_224_lite.yaml')
parser.add_argument('--num_classes', '-t', default=2, type=int, )
parser.add_argument('--device', default='cpu', type=str, )
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
print("Start 1")
args = parser.parse_args()
config = get_config(args)

begin_time = time.time()
set_seed(seed=2021)
device = args.device
epochs = args.warm_epoch + args.end_epoch
print("Getting labels and images path")
train_csv = args.csv_dir_train
df_train = pd.read_csv(train_csv)
train_imgs, train_masks = args.imgs_train_path, args.labels_train_path
#--------------------------
#import os 
#assert os.path.exists("datasets/ISIC-2017_Training_Data/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000.jpg")
#ok
#--------------------------
#print("questo è train_imgs", train_imgs) ok
# NEL FILE CSV MANCA .JPG A IMAGE_NAME!!!!
print("getting images")
train_imgs = [''.join([train_imgs, '/', i.replace('.jpg', '.jpg')]) for i in df_train['image_name']]
#print("questo è train_imgs", train_imgs) #ok
train_masks = [''.join([train_masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in df_train['image_name']]
#ORIGINAL:
#train_masks = [''.join([train_masks, '/', i.replace('.png', '_segmentation.png')]) for i in df_train['image_name']]
#provo ad aprire
#--------------------
# print("TRAIN IMGS: ",train_imgs)
# print("TRAIN masks: ",train_masks)
#--------------------
#print("TRAIN MASKS: ",train_masks) OK
print("got images")
print("taking val imgs and masks path")
df_val = pd.read_csv(args.csv_dir_val)
val_imgs, val_masks = args.imgs_val_path, args.labels_val_path
#--------------------

#--------------------
print("getting val")
val_imgs = [''.join([val_imgs, '/', i]) for i in df_val['image_name']]
## val_masks = [''.join([val_masks, '/', i]) for i in df_val['image_name']]
val_masks = [''.join([val_masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in df_val['image_name']]
#--------------------
# print(val_imgs)
# print(val_masks)
#--------------------
print("getting all training images and everything needed for the train")
#---- ORIGINAL IMPLEMENTATION --- to run on server
# imgs_train = [cv2.imread(i)[:, :, ::-1] for i in train_imgs]
# masks_train = [cv2.imread(i)[:, :, 0] for i in train_masks]
# imgs_val = [cv2.imread(i)[:, :, ::-1] for i in val_imgs]
# masks_val = [cv2.imread(i)[:, :, 0] for i in val_masks]
#---MODIFICATION DUE TO SETUP --- ONLY ON LOCAL
import cv2

# Funzione per caricare e ridimensionare le immagini
def load_and_resize_images(file_paths, target_size):
    print("in load_and_resize images")
    images = []

    for file_path in file_paths:
        #print("dentro il for")
        img = cv2.imread(file_path)[:, :, ::-1]  # Carica e converte in RGB
        img_resized = cv2.resize(img, target_size)  # Ridimensiona l'immagine
        images.append(img_resized)
    print("Done")
    return images

# Esempio di utilizzo
target_size = (512, 512)  # Specifica la dimensione desiderata
imgs_train = load_and_resize_images(train_imgs, target_size)
print("imgs_train ok")
imgs_val = load_and_resize_images(val_imgs, target_size)
print("imgs_val ok")
masks_train=load_and_resize_images(train_masks,target_size)
print("mask_train ok")
masks_val=load_and_resize_images(val_masks,target_size)
print("masks_val ok")


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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=False, num_workers=8,
                          drop_last=True, )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=False, num_workers=8, )
    best_acc = 0
    print("Start inside train function")
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
            f.close()
            t.update(1)
            
    write_options(model_savedir, args, best_acc)
    print('Done!')


if __name__ == '__main__':
    print("Starting (__main__)...")
    model = Vit(config, img_size=args.img_size, num_classes=args.num_classes)
    print("Model created (vit)")
    print("Charging config file")
    model.load_from(config)
    print("Charged config file")
    print("Starting training....")
    train(model, 'CoTrFuse_ISIC')
    print("...Training over")
