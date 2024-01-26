import math
from lmfit import Model
from lmfit import Minimizer, Parameters, minimize, report_fit, Parameter
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import lmfit

def beckmann(halfway_vector, alpha):
    # wi 是lighting和p点的连线，wo是点p和视角之间的向量
    # n是镜面法向，alpha是粗糙度
    alpha_square = alpha * alpha
    item1 = math.exp(-math.tan(halfway_vector) / alpha_square)
    item2 = math.pi * alpha_square * (math.cos(halfway_vector) * math.cos(halfway_vector) * math.cos(halfway_vector) * math.cos(halfway_vector))
    return item1 / item2

def calculate_factors(sample_xl, pos):
    wi = [np.array(a-pos) for a in sample_xl]
    return wi

def calculate_costheta_(wi, light_n):
    costheta_ = []
    for i in range(len(wi)):
        temp = (wi[i]*light_n[i]) / (np.linalg.norm(wi[i])*np.linalg.norm(light_n[i]))
        costheta_.append(temp)
    costheta_ = np.array(costheta_)
    return costheta_
def calculate_costheta(wi, pos_n):
    costheta = []
    wi = -wi
    for i in range(len(wi)):
        temp = (wi[i]*pos_n[i]) / (np.linalg.norm(wi[i])*np.linalg.norm(pos_n[i]))
        costheta.append(temp)
    costheta_ = np.array(costheta)
    return costheta_
def calculate_halfway(wi, wo):
    halfway_vector = []
    for i in range(len(wi)):
        temp = np.array((wi[i]+wo)/np.linalg.norm(wi[i]+wo))
        halfway_vector.append(temp)
    return halfway_vector
def cal_lumi(sample_xl, sample_nl, wo, pos, light_area, alpha, pos_n):
    # sample_xl 光源表面采样点
    # sample_nl 光源表面采样点法向
    # wo 出射角度
    # pl 样本表面位置点
    lumi = 0
    wi = calculate_factors(sample_xl, pos)
    halfway_vector = calculate_halfway(wi, wo)
    costheta_ = calculate_costheta_(wi, sample_nl)
    costheta = calculate_costheta(wi, pos_n)
    pdf_light = 1/light_area
    for i in range(len(wo)):
        lumi += beckmann(halfway_vector[i], alpha) * costheta * costheta_ / (np.linalg.norm(sample_xl[i] - pos_n))**2
    lumi *= pdf_light
    return lumi
def LM_fitting(lumitexel, model_out, random_para, sample_xl, sample_nl, wo, pos, light_area, alpha, pos_n):
    # 首先得到100组深度学习部分输出的lumitexel size为 [10240, 1]
    # lumitexel是某一光源发最大光时 p点对相机的反射幅度
    # model_out是深度学习部分输出的lumitexel
    # random_para是随机渲染参数，其中包括sampler表面点p和粗糙度alpha，与model_out一一对应
    gmodel = Model(cal_lumi, independent_vars=['alpha'])
    result = gmodel.fit(lumitexel, sample_xl, sample_nl, wo, pos, light_area, pos_n, alpha=Parameter('alpha', value=0.00001))
    report_fit(result.params)








