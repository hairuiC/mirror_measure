import torch
from torch import nn
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

class Lumi_model(nn.Module):
    def __init__(self, n_lightPattern):
        super(Lumi_model, self).__init__()
        # self.input = input
        # height, width = (self.input).shape
        self.n_lightPattern = n_lightPattern
        self.encoder = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=20, stride=20),
            # nn.BatchNorm2d(10240),
            # nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1, out_channels=self.n_lightPattern, kernel_size=10240),
            nn.BatchNorm2d(self.n_lightPattern),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=1, out_channels=n_lightPattern, kernel_size=(2560, 1600)),

        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.n_lightPattern, out_features=128),
            nn.BatchNorm1d(128),

            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),

            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),

            nn.Linear(in_features=2048, out_features=4096),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.4),

            nn.Linear(in_features=4096, out_features=10240),
            nn.Sigmoid()
        )

    def forward(self, input):
        print('=================', input.shape)
        input = input.to(torch.float32)
        input = self.encoder(input)
        print('-------------------encodershape', input.shape)
        input = input.view(-1)
        input = input.unsqueeze(0)
        input = self.decoder(input)
        return input





#
# class Encoder(nn.Module):
#
#     def __int__(self, input, n_lightPattern):
#         super(Encoder, self).__init__()
#         self.input = input
#         height, width = (self.input).shape
#         self.n_lightPattern = n_lightPattern
#         self.model = nn.Sequential(
#             nn.Conv1d(in_channels=width, out_channels=128, kernel_size=width),
#             # nn.Conv2d(in_channels=1, out_channels=n_lightPattern, kernel_size=(2560, 1600)),
#             nn.BatchNorm2d(n_lightPattern),
#             nn.ReLU(inplace=True)
#         )
#
#
#     def forward(self, input):
#         return self.model(input)
#         # 1 * 16个输出
#
#     def getLightPattern(self):
#         pass
#
#     def initialize(self):  # 初始化模型参数
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 a = np.sqrt(6 / (self.neural_num + self.neural_num))  # Xavier初始化方法
#                 tanh_gain = nn.init.calculate_gain('tanh')
#                 a *= tanh_gain
#                 nn.init.uniform_(m.weight.data, -a, a)
#
#
# class lumitexelDecoder(nn.Module):
#     def __int__(self):
#         super(lumitexelDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(16, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, 2560*1600),
#             nn.LeakyReLU()
#         )
#     def forward(self, input):
#         return self.model(input)
#
#     def initialize(self):  # 初始化模型参数
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 a = np.sqrt(6 / (self.neural_num + self.neural_num))  # Xavier初始化方法
#                 tanh_gain = nn.init.calculate_gain('tanh')
#                 a *= tanh_gain
#                 nn.init.uniform_(m.weight.data, -a, a)
#
# class normalDecoder(nn.Module):
#     #-------------------------------------
#     #
#     #   法向量，非线性解码器前后需要两个归一化操作
#     #
#     #-------------------------------------
#     def __int__(self):
#         super(normalDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(16, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 3),
#             nn.LeakyReLU()
#         )
#     def forward(self, input):
#         pass
#
# class geoDecoder():
#     def __init__(self):
#         super(geoDecoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 3),
#             nn.LeakyReLU(),
#         )
#
#     def forward(self, input):
#         pass



