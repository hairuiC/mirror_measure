# encoding: utf-8
# module torch._C
# from /home/jing/.conda/envs/nm/lib/python3.8/site-packages/torch/_C.cpython-38-x86_64-linux-gnu.so
# by generator 1.147
# no doc

# imports
import torch
import torch.functional as F
import torch.optim as optim
import numpy as np
import argparse
from model.utils import *
from model.dataset import *
from model.model import *
from torch.utils.data import DataLoader
import math


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a roughness network.')

    # parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--img_path', help='Path to file containing training annotations (see readme)')
    # parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--test_path', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--batch_size', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs_nums', help='Number of epochs', type=int, default=100)
    parser.add_argument('--learning_rate', default=1e-3)


    parser = parser.parse_args(args)

    #---------------------------先做相机标定和图像正畸

    #---------------------------读取数据集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_train = Lumitexel_Dataset(
        img_path= parser.img_path,
        split=0,
        transform=None
    )
    dataLoader = DataLoader(dataset_train, batch_size=parser.batch_size, num_workers=1)
    criterion =


    #---------------------------设置优化器、模型、loss函数

    rough_model = lumitexelDecoder()
    optimizer = optim.RMSprop(params=rough_model.parameters(), lr=parser.learning_rate)

    #---------------------------开始训练
    for epoch in range(parser.epoch_nums):
        rough_model.train()
        epoch_loss = []

        for iter_num, data in enumerate(dataLoader):
            imgs, _ = data
            optimizer.zero_grad()
            outputs = rough_model.forward(imgs)
            loss = criterion(outputs, label)
            epoch_loss.append(loss)

