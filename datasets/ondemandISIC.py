# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader

# class OnDemandISIC(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.image_paths = os.listdir(data_dir)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.data_dir, self.image_paths[idx])
#         with Image.open(image_path) as image:
#             image = image.convert('RGB')
#             tensor_image = torch.tensor(image)
#         return tensor_image
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OnDemandISIC2017(Dataset):
    def __init__(self, csv, imgs_path, labels_path, transform, training=True):
        self.transform = transform
        self.training = training
        self.df = pd.read_csv(csv)
        self.imgs_path,self.labels_path = imgs_path, labels_path
        # Inizializza imgs_path e labels_path come attributi dell'oggetto
        self.imgs_path = [''.join([self.imgs_path, '/', i.replace('.jpg', '.jpg')]) for i in self.df['image_name']]
        self.labels_path = [''.join([self.labels_path, '/', i.replace('.jpg', '_segmentation.png')]) for i in self.df['image_name']]

    def __getitem__(self, index):
        img=cv2.imread(self.imgs_path[index])[:,:,::-1]
        label=cv2.imread(self.labels_path[index])[:,:,0]
        data=self.transform(image=img,mask=label)
        return data['image'],(data['mask']/255).long()

    def __len__(self):
        return len(self.df)


# Define your transformations
def for_train_transform():
    desired_size = 512
    train_transform = A.Compose([
        A.Resize(width=desired_size, height=desired_size),
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.1, p=0.5),
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
    A.Resize(width=512, height=512),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)