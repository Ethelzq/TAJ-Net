#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:08:19 2021
@author: Ethel   School:ECNU   Email:52181214003@stu.ecnu.edu.cn
"""
import os
from PIL import Image
import torch
from spectral import principal_components
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import torchvision
import cv2
import glob
import sys
from data.argument import Transform
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class lung_Dataset_3d(data.Dataset):

    def __init__(self, img_dirs, mode='train', image_shape=(40,224,224), transform=None):
        self.mode = mode
        self.img_dirs = img_dirs
        self.transform = transform
        self.image_shape = image_shape

    def __getitem__(self, index):
        img_dir = self.img_dirs[index]
        img = np.load(img_dir)
        img = img.astype(np.float32)  # / 255.
        if self.image_shape[0] == 3:
            img = img[:,:,8:38:10]

        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / img_std
        spectral = np.zeros(self.image_shape[0],np.float32)
        # print(img.shape)
        for i in range(img.shape[2]):
            img[:, :, i] = ((img[:, :, i] - np.min(img[:, :, i])) /
                            (np.max(img[:, :, i]) - np.min(img[:, :, i])))  # * 255
            # print(img[:,:,i])
            spectral[i] = img[111,111,i]
        if self.image_shape[0] == 3:
            spec_mean_data = spectral
            band = range(0, self.image_shape[0])
        else:
            spec_mean_data = spectral[4:20]
            band = range(0, self.image_shape[0])
        spec_mean = spec_mean_data.mean()
        # img = np.expand_dims(img, axis=1)# not for use
        # # show the spectral curve
        # plt.plot(band, spectral)
        # plt.xlabel(str(img_dir))
        # img_name = img_dir.split('/')[-1]
        # plt.savefig('/home/zq/Documents/TSC-Net-2D-3D/senet_curve/' + str(img_name) + '.png', bbox_inches='tight')
        # plt.show()
        # plt.close()
        if self.mode == 'test':
            if self.transform:
                img = self.transform(img)
            return img
        else:
            if 'tumor' in img_dir:
                label = 0
            elif 'hyper' in img_dir:
                label = 1
            else:
                label = 2
            example = img, np.array(label)
            if self.transform:
                img, label = self.transform(example)
            return img, label, index, spec_mean

    def __len__(self):
        return len(self.img_dirs)


class lung_Dataset(data.Dataset):
    def __init__(self, img_dirs, mode='train', transform=None):
        self.mode = mode
        self.img_dirs = img_dirs
        self.transform = transform
        # normalize = T.Normalize(mean=[0.5], std=[0.5])

    def __getitem__(self, index):
        img_dir = self.img_dirs[index]
        # img = cv2.imread(img_dir)[:,:,::-1] # for contour_detection
        # img = cv2.resize(img,(300,300))
        img = np.load(img_dir)
        # img= img[:, :, 8:38:10]
        img = img.astype(np.float32)  # / 255.
        img = np.expand_dims(img, axis=1)
        if self.mode == 'test':
            if self.transform:
                img = self.transform(img)
            return img
        else:
            if 'tumor' in img_dir:
                label = 0
            elif 'hyper' in img_dir:
                label = 1
            else:
                label = 2
            example = img, np.array(label)
            # print(label)Only 2-D images (grayscale or color) are supported, when providing a callable `inverse_map`.
            if self.transform:
                img, label = self.transform(example)
            return img, label

    def __len__(self):
        return len(self.img_dirs)


def build_dataset_3d(data_root: str, split=0.1, mode='train', image_shape=(40, 224, 224), is_argument=True):
    datadir = glob.glob(data_root + '*/*.npy')
    # print(datadir[:5])
    X_train, X_val = train_test_split(datadir, test_size=0.1, random_state=1)
    if mode == 'val':
        val_transform = Transform(size=image_shape, sigma=-1., affine=False)
        val_db = lung_Dataset_3d(X_val, mode=mode, image_shape=image_shape, transform=val_transform)
        return val_db
    elif mode == 'test':
        test_transform = Transform(size=image_shape, sigma=-1., affine=False)
        test_db = lung_Dataset_3d(datadir, mode=mode, image_shape=image_shape, transform=test_transform)
        return test_db
    else:
        if is_argument:
            print('train with argument!!!!')
            train_transform = Transform(size=image_shape, sigma=-1., affine=True)
        else:
            print('train without argument!!!!')
            train_transform = Transform(size=image_shape, sigma=-1., affine=False)
        train_db = lung_Dataset_3d(X_train, mode=mode, image_shape=image_shape, transform=train_transform)
        # if True in np.isnan(train_db):
        #     print('there is nan in the train_dataset!!!!')
        return train_db


def build_dataset(data_root: str, split=0.1, mode='train', image_shape=(224, 224), is_argument=False):
    datadir = glob.glob(data_root + '/*/*.npy')  # contour: jpg
    # print(datadir[:5])
    X_train, X_val = train_test_split(datadir, test_size=0.1, random_state=1)
    if mode == 'val':
        val_transform = Transform(size=image_shape, sigma=-1., affine=False)
        val_db = lung_Dataset(X_val, mode=mode, transform=val_transform)
        return val_db
    elif mode == 'test':
        val_transform = Transform(size=image_shape, sigma=-1., affine=False)
        val_db = lung_Dataset(datadir, mode=mode, transform=val_transform)
    else:
        if is_argument:
            print('train with argument!!!!')
            train_transform = Transform(size=image_shape, sigma=-1., affine=True)
        else:
            print('train without argument!!!!')
            train_transform = Transform(size=image_shape, sigma=-1., affine=False)
        train_db = lung_Dataset(X_train, mode=mode, transform=train_transform)

        return train_db


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    datadir = './train_data_hyperspectral/'
    train_db = build_dataset(datadir, mode='val')
    img, label = train_db[1]
    # print(label)
    # plt.imshow(img)
    print(img.shape, label)