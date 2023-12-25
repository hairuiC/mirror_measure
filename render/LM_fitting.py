import math

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
def brdf():
    pass

def cal_deriv(params, render_lumi, para_index):
    # render_lumi: list -->10240 * 1 * num
    para_back1 = params.copy()
    para_back2 = params.copy()
    para_back1[para_index, 0] += 0.0000001
    para_back2[para_index, 0] -= 0.0000001



def LM_fitting(brdf_func, model_out, random_para):
    # 首先得到100组深度学习部分输出的lumitexel size为 [10240, 1]
    # lumitexel是某一光源发最大光时 p点对相机的反射幅度
    # model_out是深度学习部分输出的lumitexel
    # random_para是随机渲染参数，其中包括sampler表面点p和粗糙度alpha，与model_out一一对应
    random_x = np.random.uniform(0.001, 0.1, 100)
    x_data = [math.log2(x) for x in random_x]
    popt, pcov = curve_fit(brdf, x_data, model_out, bounds=(0, [3., 1., 0.5]))
