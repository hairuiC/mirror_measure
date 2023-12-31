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


def find_corners(img, chess_col, chess_row, sav_path, is_save=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_col, chess_row), None)
    # 终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
    if ret == True:
        cv2.drawChessboardCorners(img, (chess_col, chess_row), corners2, ret)
        if is_save is True:
            cv2.imwrite(sav_path, img)
    return ret, corners2


def calibration(chess_path, chess_col, chess_row, fx_val=1.0, fy_val=1.0, ):
    h = 0
    w = 0
    # 准备对象点
    dic_distortion = {}
    obj_p = np.zeros((9 * 6, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # 用于存储所有图像的队形点和图像点的数组
    obj_pts = []  # 真实世界的3d点
    img_pts = []  # 图像中的2d点
    get_path = chess_path + "/*.jpg"
    images = glob.glob(get_path)
    for image in images:
        print(image.split("/")[1] + "读入成功！")
        img = cv2.imread(image)
        img = cv2.resize(img, None, fx=fx_val, fy=fy_val)
        (h, w) = img.shape[:2]
        save_draw_chess_path = chess_path + "corner_" + str(image.split("/")[1])
        ret, sub_corner = find_corners(img, chess_col, chess_row, sav_path=save_draw_chess_path)
        print(save_draw_chess_path + "写入成功")
        obj_pts.append(obj_p)
        img_pts.append(sub_corner)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (h, w), None, None)
    # print((h, w))


    dic_distortion['ret'] = ret
    dic_distortion['mtx'] = mtx
    dic_distortion['dist'] = dist
    dic_distortion['rvecs'] = rvecs
    dic_distortion['tvecs'] = tvecs

    save_para(dic_distortion)

    return dic_distortion


def correction(img, dic):
    # 输入path/dic
    # 读取mtx, rst 进行计算后写矫正图像
    matrix = dic["mtx"]
    dist = dic["dist"]
    (h1, w1) = img.shape[:2]
    # 对参数做处理，使得最后的输出的矫正图像去掉不必要的边缘。
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w1, h1), 1, (w1, h1))
    # 矫正
    dst = cv2.undistort(img, matrix, dist, None, newcameramtx)
    # 保存矫正图像
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


def mean_error(obj_pts, img_pts, matrix, dist, r_vecs, t_vecs):
    mean_error = 0
    for i in range(len(obj_pts)):
        img_pts2, _ = cv2.projectPoints(obj_pts[i], r_vecs[i], t_vecs[i], matrix, dist)
        error = cv2.norm(img_pts[i], img_pts2, cv2.NORM_L2) / len(img_pts2)
        mean_error += error
    return mean_error / len(obj_pts)


def save_para(dic):
    # 保存参数
    # camera_para_dict = {"ret": ret, "matrix": matrix, "dist": dist, "r_vecs": r_vecs, "t_vecs": t_vecs}
    np.save("camera_para_dict.npy", dic)

def load_para(para_path):
    para = np.load(para_path, allow_pickle=True).item()

    ret = para["ret"]
    matrix = para["matrix"]
    dist = para["dist"]
    r_vecs = para["r_vecs"]
    t_vecs = para["t_vecs"]

    return ret, matrix, dist, r_vecs, t_vecs


def main(fx_val:float, fy_val:float):
    # ###################
    chess_path = 'cali_img'
    test_img_path = '/home/jing/Documents/files'
    save_path = 'home/jing/Documents/dist_imgs'
    img_list = []
    for img_name in os.listdir(test_img_path):
        img_list.append(img_name)
    # ###################
    dic_distortion= calibration(chess_path, 9, 6, fx_val,fy_val)
    # save_para(ret, matrix, dist, r_vecs, t_vecs)
    for test_imgname in img_list:
        test_img = cv2.imread(test_img_path + os.sep + test_imgname)
        test_img = cv2.resize(test_img, None, fx=fx_val, fy=fy_val)
        correction_test_img = correction(test_img, dic_distortion)
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

    # write_obj
    B = np.array([1.262660, 0.0117442, -0.013888])
    C = np.array([1.277871, -0.002134, 1.471240])
    vector_BC = C - B
    obj_length = np.linalg.norm(vector_BC)

    # main(0.5,0.5)
    # a = tanh
    O = np.array([0.280551, 0.507936, 0.141849])
    X = np.array([-0.222432, 0.519973, 0.146457])
    Y = np.array([0.294847, 0.979378, 0.772472])
    test_patter = np.random.rand(80,128)
    file = open("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back.xml", 'w').close()
    utils.copy_file("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene.xml", "/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back.xml")
    for i in range(5):
        new_pattern = np.where((test_patter < (i+1)*0.2) & (test_patter > i*0.2),  np.ones_like(test_patter), np.zeros_like(test_patter))
        scale = obj_length / 40
        utils.write_obj(new_pattern, O, X, Y, scale, i)
    utils.del_empty("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back.xml", "/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back_new.xml")

    # test_patter[test_patter > 0.9] = 1
    # test_patter[test_patter <= 0.9] = 0
    # scale = obj_length / 40
    # utils.write_obj(test_patter, O, X, Y, scale)




# cali = utils.calibration('test_img', 'new_img')
# cali.save_newimg()