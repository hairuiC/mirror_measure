import torch
from torch import nn
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor

class Encoder(nn.Module):

    def __int__(self, n_lightPattern):
        super(Encoder, self).__init__()
        self.n_lightPattern = n_lightPattern
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_lightPattern, kernel_size=(2560, 1600)),
            nn.BatchNorm2d(n_lightPattern),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.model(input)
        # 1 * 16个输出

    def getLightPattern(self):
        pass

class lumitexelDecoder(nn.Module):
    def __int__(self):
        super(lumitexelDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2560*1600),
            nn.LeakyReLU()
        )
    def forward(self, input):
        return self.model(input)

class normalDecoder(nn.Module):
    #-------------------------------------
    #
    #   法向量，非线性解码器前后需要两个归一化操作
    #
    #-------------------------------------
    def __int__(self):
        super(normalDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
            nn.LeakyReLU()
        )
    def forward(self, input):
        pass

class geoDecoder():
    def __init__(self):
        super(geoDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        pass



