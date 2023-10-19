from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import torch
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
class ISIC2017(Dataset):
    #mask=label
    #images and mask are csv
    def __init__(self,csv,imgs_path,labels_path,transform,training=True):
        self.transform=transform
        if training:
            self.df = pd.read_csv(csv)
            self.images, self.masks = imgs_path, labels_path
            # NEL FILE CSV MANCA .JPG A IMAGE_NAME!!!!
            print("getting images")
            self.images = [''.join([self.images, '/', i.replace('.jpg', '.jpg')]) for i in self.df['image_name']]
            self.masks = [''.join([self.masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in self.df['image_name']]
        else:
            print("taking val imgs and masks path")
            self.df = pd.read_csv(csv)
            self.images, self.masks = imgs_path, labels_path
            print("getting val")
            self.images = [''.join([self.images, '/', i]) for i in self.df['image_name']]
            self.masks = [''.join([self.masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in self.df['image_name']]
                    
        
    def __getitem__(self, index):
        img=cv2.imread(self.images[index])
        label=np.array(self.masks[index])
        img = cv2.resize(img, (512,512))
        if self.transform:
            img=self.transform(image=img,mask=label)
        return img,label
    
    def __len__(self):
        return len(self.images)
    
def for_train_transform():
    desired_size = 512
    train_transform = A.Compose([
        A.Resize(width=desired_size, height=desired_size),
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.1,
            p=0.5
        ),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=100, val_shift_limit=80),
        A.GaussNoise(),
        A.OneOf([
            A.ElasticTransform(),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0)
        ]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform 

test_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)