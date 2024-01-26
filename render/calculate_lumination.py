import numpy as np
import cv2
import math
import scipy.io as scio
from model.utils import *


class lumination_func():
    def __init__(self, x_pos, light_pos, cam_pos, normal_pos, normal_light, scale, light_width, light_height, transformer, rotation):
        self.x_pos = x_pos
        self.light_pos = light_pos #光源左上角
        self.scale = scale
        self.light_width = light_width #光源尺寸
        self.light_height = light_height
        self.cam_pos = cam_pos
        self.n_p = normal_pos
        self.n_l = normal_light
        self.transformer = transformer
        self.rotation = rotation
    def beckmann(self , wi, wo, alpha):
        # wi 是lighting和p点的连线，wo是点p和视角之间的向量
        # n是镜面法向，alpha是粗糙度
        halfway_vector = (wi + wo) / np.linalg.norm(wi + wo)
        alpha_square = alpha*alpha
        item1 = math.exp(-math.tan(halfway_vector)/alpha_square)
        item2 = math.pi * alpha_square * (math.cos(halfway_vector) * math.cos(halfway_vector) * math.cos(halfway_vector) * math.cos(halfway_vector))
        return item1/item2
    def cal_wi_np(self, sample_pos):
        view_dir = self.cam_pos - self.x_pos
        view_dir = np.linalg.norm(view_dir)

        wi = sample_pos - self.x_pos
        wi = np.linalg.norm(wi)

        wiNp = np.dot(wi, self.n_p)
        neg_wiNl = np.dot(-wi, self.n_l)

        return wiNp, neg_wiNl
    def sampling(self, intensity_w:int, intensity_h:int):
        # 面光源采样 10*10
        sampling_light_point = []
        x_index, y_index = (np.linspace(0, self.light_width, intensity_w),
                            np.linspace(0, self.light_height, intensity_h))
        for y in y_index:
            for x in x_index:
                sampling_light_point.append(np.array([transformation(self.rotation, self.transformer, x), transformation(self.rotation, self.transformer, y)]))

        return sampling_light_point
    def compute_formFactor(self, light:list):
        # light参数代表光源，按顺序包含光源v0-v3 四个顶点

        luminare = 1/(2*math.pi)
        temp_sum = 0
        Y_0 = np.array(self.x_pos - light[0])
        Y_1 = np.array(self.x_pos - light[1])
        Y_2 = np.array(self.x_pos - light[2])
        Y_3 = np.array(self.x_pos - light[3])
        y_axis = np.array([0, 1, 0])
        for i in range(4):
            # 4个edge
            n_ij = np.multiply(light[i%3], light[i%3+1])
            n_ij = np.linalg.norm(n_ij)
            temp_sum += 1/np.pi * (n_ij*y_axis)*(np.arccos(np.dot(light[i%3], light[i%3+1]))/2)
        return temp_sum
        # while True:
        #     np.dot(self.n_p, np.multiply())

    def cal_brdf(self, p_pos, view_pos, n_light, n_pos, phi, alpha, lumina_light:list):
        # p_pos代表点p位置，light_pos代表光源位置，view_pos代表相机位置
        # n_light代表光源朝向（法向），n_pos代表p点法向，phi代表光强角度分布

        sum_lumi = 0
        for i in range(10240):

        # 每个光源 采样+计算
        # 采样 采样点和点p之间的距离、角度等
            lumi = lumina_light[i]
            temp_sample = 0
            sampling_light_point = self.sampling(intensity_w=10, intensity_h=10)
            radiance = 1/100
            for light_sample in sampling_light_point:
                lumination = self.beckmann((p_pos - light_sample), (p_pos - view_pos), alpha)
                wi_np, neg_wiNl = self.cal_wi_np(sample_pos=light_sample)
                d = p_pos - light_sample
                form_factor = radiance*((math.sqrt(d*d)) * (math.sqrt(d*d)))
                temp_sample += form_factor*wi_np*neg_wiNl*lumination/np.linalg.norm(light_sample-p_pos)
            sum_lumi += lumi * temp_sample
        return sum_lumi








