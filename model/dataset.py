import cv2
import numpy as np
import torch, torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy
import os
from torch import nn
from PIL import Image
import glob
import xml.etree.ElementTree as ET
# import utils
def no_learningConv2d(kernel_size, stride, input):
    kernel = np.ones((kernel_size, kernel_size))
    # i = 0;
    # j = 0;
    # print(i)
    H,W = input.shape
    # print(len(input[0]))
    # print(len(input[1]))
    # print(H/stride, W/stride)
    res = np.ones((int(H/stride), int(W/stride)))
    for i in range(0, int(H), stride):
        for j in range(0, int(W), stride):
            temp_value = (sum(sum(np.multiply(kernel, input[i:i+stride, j:j+stride]))))/400
            # print(((i+stride)/stride)-1, ((j + stride)/stride)-1)
            res[int(((i+stride)/stride)-1)][int(((j + stride)/stride)-1)] = temp_value
    return res



# 假设lumitexel(单独最大光照情况的渲染图)已经存储在my_scene/render_lumitexel中
def create_trainingset(N, lumi_filepath):
    # beckmann 方程重建
    # 1. 随机选择空间点p(坐标变换)
    # 2. 以beckmann方程为基准随机生成alpha
    # 2. p点的像素值取出作为lumitexel
    filenames = glob.glob(lumi_filepath)
    width, height = (cv2.imread(filenames[0])).shape
    train_set = []
    for i in range(N):
        lumitexel = []
        for files in filenames:
            img = cv2.imread(files)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_array = np.array(img)
            [X, Y] = np.random.uniform(0, 1, 2)
            X = int(X * width)
            Y = int(Y * height)
            lumitexel.append(img_array[X, Y])
        train_set.append(lumitexel)
    return train_set




class Lumitexel_Dataset(Dataset):
    def __init__(self, img_path, split, transform):
        self.img_path = img_path
        self.split = split
        self.filelist = []
        for filename in os.listdir(self.img_path):
            self.filelist.append(self.img_path + os.path.sep + filename)
        self.transform = transform
        # self.tempConv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=20, stride=20)
        # torch.nn.init.ones_(self.tempConv.weight)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image = Image.open(self.filelist[index]).convert('L')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageA = np.array(image)
        label = imageA
        # imageA = no_learningConv2d(kernel_size=20, stride=20, input=imageA)
        # label = no_learningConv2d(kernel_size=20, stride=20, input=label)


        if self.transform is not None:
            image = self.transform(imageA)
            label = self.transform(label)
        # image = self.tempConv(image.unsqueeze(0))
        # label = self.tempConv(label.unsqueeze(0))

        print(image.shape, label)
        # image = image.view(-1)
        # label = label.view(-1)
        return image, label
