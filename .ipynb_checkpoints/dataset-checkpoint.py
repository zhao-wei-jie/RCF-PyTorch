import torch
import cv2
import numpy as np
import os.path as osp
import PIL.Image as Image
from torch import serialization

class BSDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='data/HED-BSDS', split='test', transform=False):
        super(BSDS_Dataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            self.file_list = osp.join(self.root, 'test.lst')
        else:
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
        self.mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.split == 'train':
            img_file, label_file = self.file_list[index].split()
            label = cv2.imread(osp.join(self.root, label_file), 0)
            label = np.array(label, dtype=np.float32)
            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label > 0, label < 127.5)] = 2
            label[label >= 127.5] = 1
        else:
            img_file = self.file_list[index].rstrip()

        img = cv2.imread(osp.join(self.root, img_file))
        img = np.array(img, dtype=np.float32)
        img = (img - self.mean).transpose((2, 0, 1))

        if self.split == 'train':
            return img, label
        else:
            return img

class TTPLA_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='../ttpla/', split='test', transform=False):
        super(TTPLA_Dataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'ttpla_train.txt')
        elif self.split == 'test':
            self.file_list = osp.join(self.root, 'ttpla_val.txt')
            
        else:
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
            print(self.split,len(self.file_list))
        self.mean = np.array([62.364 ,61.025 ,56.462], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.split == 'train':
            img_file= self.file_list[index]
            label = Image.open(osp.join(self.root,'annpng_powerline', img_file.strip('\n')+'.png'))
            label=self.transf(label)
            label = np.array(label, dtype=np.float32)
            label = label[np.newaxis, :, :]
            # label[label == 0] = 0
            # label[np.logical_and(label > 0, label < 127.5)] = 2
            # label[label >= 127.5] = 1
        else:
            img_file = self.file_list[index]

        img = Image.open(osp.join(self.root, img_file.strip('\n')+'.jpg'))
        img=self.transf(img)
        img = np.array(img, dtype=np.float32)
        img = (img - self.mean).transpose((2, 0, 1))
        
        if self.split == 'train':
            return img, label
        else:
            return img
    
    def transf(self,img):
        if self.transform:
            img=self.transform(img)
        return img
