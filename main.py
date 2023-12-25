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

import model.utils
from model.utils import *
from model.dataset import *
from model.model import *
from torch.utils.data import DataLoader
from torchvision import transforms

from torchsummary import summary
import math


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a roughness network.')

    # parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--img_path', help='Path to file containing training annotations (see readme)', default="/home/jing/Documents/new1")
    # parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--test_path', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--batch_size', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=1)
    parser.add_argument('--epochs_nums', help='Number of epochs', type=int, default=100)
    parser.add_argument('--learning_rate', default=1e-3)


    parser = parser.parse_args(args)

    #---------------------------先做相机标定和图像正畸

    #---------------------------读取数据集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_transforms = transforms.ToTensor()





    dataset_train = Lumitexel_Dataset(
        img_path= parser.img_path,
        split=0,
        transform=data_transforms
    )
    dataLoader = DataLoader(dataset_train, batch_size=parser.batch_size, num_workers=1, drop_last=True)



    #---------------------------设置优化器、模型、loss函数

    rough_model = Lumi_model(n_lightPattern=2)
    params = list(rough_model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))


    # summary(rough_model, input_size=(2, 1, 10240), batch_size=1)
    # rough_model.to(device)
    print(rough_model)
    optimizer = optim.RMSprop(params=rough_model.parameters(), lr=parser.learning_rate)

    #---------------------------开始训练
    for epoch in range(parser.epochs_nums):
        print(epoch)
        rough_model.eval()
        epoch_loss = []

        for iter_num, data in enumerate(dataLoader):
            imgs, _ = data
            optimizer.zero_grad()
            print('inputshape:', imgs.shape)
            outputs = rough_model.forward(imgs)
            print('//////////////', outputs.shape)
            print(outputs)
            weight = rough_model.parameters()
            criterion = model.utils.criterion(input=outputs, GT=imgs, epsilon=0.005, labda=0.3, weight=weight)
            loss = criterion.DAE_Loss()
            epoch_loss.append(loss)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #------------------------------训练结束 获取lighting pattern

    lightingpattern, light_grad = get_lightingPattern(rough_model)


if __name__ == '__main__':
    main()





































