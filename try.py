import math
import os

import numpy as np
from model import utils
import random
# a = []
# a = math.log(0.006, 10)
# b = math.log(0.5, 10)
#
# print(np.random.uniform(a, b, 100))

import cv2
import numpy as np
import glob

def main(fx_val:float, fy_val:float):
    # ###################
    chess_path = 'cali_img'
    test_img_path = '/home/jing/Documents/files'
    save_path = 'home/jing/Documents/dist_imgs'
    img_list = []
    for img_name in os.listdir(test_img_path):
        img_list.append(img_name)
    # ###################
    dic_distortion= utils.calibration(chess_path, 9, 6, fx_val,fy_val)
    # save_para(ret, matrix, dist, r_vecs, t_vecs)
    for test_imgname in img_list:
        test_img = cv2.imread(test_img_path + os.sep + test_imgname)
        test_img = cv2.resize(test_img, None, fx=fx_val, fy=fy_val)
        correction_test_img = utils.correction(test_img, dic_distortion)
        cv2.imwrite(save_path+os.sep+test_imgname, correction_test_img)
    # cv2.imshow("test_img", test_img)
    # cv2.imshow("correction_test_img", correction_test_img)
    # m_error = mean_error(obj_pts, img_pts, matrix, dist, r_vecs, t_vecs)
    # print("重投影误差：", m_error)
    # cv2.waitKey(0)

def tanh(x):
    return math.tanh(x)
if __name__ == '__main__':
    # mtx, dist, scale = calibration("/home/jing/PycharmProjects/heliostat_measure/cali_img", 9, 6, fx_val=0.7, fy_val=0.7)
    # print(mtx, dist)
    main(0.7, 0.7)
    # write_obj
    # ----------------------------------------------------------- temp
    # B = np.array([1.262660, 0.0117442, -0.013888])
    # C = np.array([1.277871, -0.002134, 1.471240])
    # vector_BC = C - B
    # obj_length = np.linalg.norm(vector_BC)
    #
    # # main(0.5,0.5)
    # # a = tanh
    # O = np.array([0.280551, 0.507936, 0.141849])
    # X = np.array([-0.222432, 0.519973, 0.146457])
    # Y = np.array([0.294847, 0.979378, 0.772472])
    # test_patter = np.random.rand(80,128)
    # file = open("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back.xml", 'w').close()
    # utils.copy_file("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene.xml", "/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back.xml")
    # for i in range(5):
    #     new_pattern = np.where((test_patter < (i+1)*0.2) & (test_patter > i*0.2),  np.ones_like(test_patter), np.zeros_like(test_patter))
    #     scale = obj_length / 40
    #     utils.write_obj(new_pattern, O, X, Y, scale, i)
    # utils.del_empty("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back.xml", "/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back_new.xml")
    #--------------------------------------------------------------- temp end
    # test_patter[test_patter > 0.9] = 1
    # test_patter[test_patter <= 0.9] = 0
    # scale = obj_length / 40
    # utils.write_obj(test_patter, O, X, Y, scale)




# cali = utils.calibration('test_img', 'new_img')
# cali.save_newimg()