#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader2.py
# @Author: Jehovah
# @Date  : 18-7-30
# @Desc  : 


"""
load data
"""
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio

IMG_EXTEND = ['.jpg', '.JPG', '.jpeg', '.JPEG',
              '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
              ]


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTEND)


def mat_process(img_fl):
    """
    process mat, 11 channel to 8 channel
    :param img_fl:
    :return:
    """
    img_fl = img_fl.astype(np.float32)
    temp = img_fl
    lists = []
    refen = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 6), (7, 9), (8, 8), (10, 10)]
    for item in refen:
        aa, bb = item
        if aa == bb:
            ll = temp[aa, :, :]
        else:
            ll = temp[aa, :, :] + temp[bb, :, :]
            ll = np.where(ll > 1, 1, ll)
        lists.append(ll.reshape(1, ll.shape[0], ll.shape[1]))
    parsing = np.concatenate(lists, 0)

    return parsing


def make_dataset(dir, file):
    imgA = []
    imgB = []

    file = os.path.join(dir, file)
    fimg = open(file, 'r')
    for line in fimg:
        line = line.strip('\n')
        line = line.rstrip()
        word = line.split("||")
        imgA.append(os.path.join(dir, word[0]))
        imgB.append(os.path.join(dir, word[1]))

    return imgA, imgB


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(data.Dataset):
    def __init__(self, opt, isTrain=0, transform=None, return_paths=None, loader=default_loader):
        super(MyDataset, self).__init__()
        self.opt = opt
        if isTrain==0:
            self.mode = 'train'
        else:
            self.mode = "test"
        imgs = os.listdir(os.path.join(opt.dataroot, self.mode, 'photos'))
        # imgs = os.listdir(os.path.join(opt.dataroot, 'image', self.mode))
        # imgs = make_dataset(self.opt.dataroot, self.opt.datalist)
        # imgs_test = make_dataset(self.opt.dataroot, self.opt.datalist_test)
        if len(imgs) == 0:
            raise (RuntimeError(
                "Found 0 images in: " + self.opt.dataroot + dir + "\n" "Supported image extensions are: " + ",".join(
                    IMG_EXTEND)))

        self.isTrain = isTrain
        self.imgs = imgs
        # self.imgs_test = imgs_test
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):

        imgth = self.imgs[index]
        # A = Image.open(os.path.join(os.path.join(self.opt.dataroot, 'image', self.mode), imgth)).convert('RGB')
        # B = Image.open(os.path.join(os.path.join(self.opt.dataroot, 'sketch', self.mode), imgth)).convert('RGB')
        A = Image.open(os.path.join(os.path.join(self.opt.dataroot,self.mode, "photos"), imgth)).convert('RGB')
        B = Image.open(os.path.join(os.path.join(self.opt.dataroot,self.mode, "sketch"), imgth)).convert('RGB')
        # mask = Image.open(os.path.join(os.path.join(self.opt.dataroot,self.mode, "mask"), imgth[:-3] + 'png')).convert('L')
        if self.isTrain == 0:
            w, h = A.size
            pading_w = (self.opt.loadSize - w) / 2
            pading_h = (self.opt.loadSize - h) / 2
            padding = transforms.Pad((int(pading_w), int(pading_h)), fill=0, padding_mode='constant')
            # padding = transforms.Pad((pading_w, pading_h), padding_mode='edge')
            i = random.randint(0, self.opt.loadSize - self.opt.fineSize)
            j = random.randint(0, self.opt.loadSize - self.opt.fineSize)
            A = self.process_img(A, i, j, padding)
            B = self.process_img(B, i, j, padding)
        else:
            w, h = A.size
            pading_w = (self.opt.fineSize - w) / 2
            pading_h = (self.opt.fineSize - h) / 2
            padding = transforms.Pad((int(pading_w), int(pading_h)), fill=0, padding_mode='constant')
            # padding = transforms.Pad((pading_w, pading_h), padding_mode='edge')
            A = padding(A)
            B = padding(B)

            A = transforms.ToTensor()(A)
            B = transforms.ToTensor()(B)
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        return {'A': A, 'B': B, 'A_path': os.path.join(os.path.join(self.opt.dataroot, 'image', self.mode), imgth),
                'B_path': os.path.join(os.path.join(self.opt.dataroot, 'sketch', self.mode), imgth)}

    def __len__(self):
        return len(self.imgs)

    def process_img(self, img, i, j, padding):
        img = padding(img)
        img = img.crop((j, i, j + self.opt.fineSize, i + self.opt.fineSize))
        img = transforms.ToTensor()(img)
        # if self.isTrain == 0:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        # if len(img.shape) == 4:
        #     img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        # else:
        #     img = transforms.Normalize((0.5,), (0.5,))(img)
        return img


if __name__ == '__main__':
    pass
