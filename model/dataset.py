import cv2
import torch, torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy
import os

class Lumitexel_Dataset(Dataset):
    def __init__(self, img_path, split, transform):
        self.img_path = img_path
        self.split = split
        self.filelist = []
        for filename in os.listdir(self.img_path):
            self.filelist.append(self.img_path + os.path.sep + filename)
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):

        image = cv2.imread(self.img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image
