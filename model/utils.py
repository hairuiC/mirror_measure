import torch
import numpy as np
import math
import torch.nn.init
import glob
import cv2
import os.path
import os


class random_sampling():
    def __init__(self):
        pass
    def Ro_sampling(self, num):
        #---------------
        # 0, 1范围内随机采样反射率
        return np.random.uniform(size=num)

    def alpha_sampling(self, low, high, num):
        #---------------
        # 0.006 - 0.5 对数尺度范围内均匀随机采样粗糙度 粗糙度是表面法向分布的参数
        return np.random.uniform(math.log(low, 10), math.log(high, 10), num)

def init_weights(model):
    pass

class calibration():
    def __init__(self, img_path, save_pth):
        # self.boardwidth = boardwidth
        # self.boardheigh = boardheight
        self.img_path = img_path
        self.save_pth = save_pth
        # self.Matrix =

    def calibrating(self):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        objp = 2.5 * objp  # 打印棋盘格一格的边长为2.6cm
        obj_points = []  # 存储3D点
        img_points = []  # 存储2D点
        images = glob.glob("cali_img/*.jpg")  # 黑白棋盘的图片路径

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
                if [corners2]:
                    img_points.append(corners2)
                    print()
                # else:
                #     img_points.append(corners)
                # cv2.drawChessboardCorners(img, (9, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
                # cv2.imshow('temp', img)
                # cv2.waitKey(0)
        _, matrix, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)

        return matrix, dist

    def correction(self, img, matrix, dist):
        # img = cv2.imread(self.img_path)
        (h1, w1) = img.shape[:2]

        newcameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist, (h1, w1), 1, (w1, h1))

        dst = cv2.undistort(img, matrix, dist, None, newcameraMatrix)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x+w]
        return dst

    def contour_demo(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        ref, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        return contours

    def perspective(self, img):
        contours = self.contour_demo(img)
        contour = contours[0]


    def save_newimg(self):
        if (not os.path.exists(self.save_pth)):
            os.mkdir(self.save_pth)

        filenames = os.listdir(self.img_path)
        for filename in filenames:
            if not os.path.isdir(filename):
                # file = open(self.img_path+filename)
                img = cv2.imread(self.img_path + '/' + filename)
                matrix, dist = self.calibrating()
                newimg = self.correction(img, matrix, dist)
                cv2.imwrite(self.save_pth+filename, newimg)

class correction():
    def __init__(self):
        pass

    def correction(self):
        pass

    def save_img(self):
        pass


def inverse_color(grey_img):
    # 反色函数
    return cv2.bitwise_not(grey_img)
    # return Inverse_frame_gray

def img_xor(img1, img2):
    # 图像异或，主要用来展示不同进光量摄像头拍摄图像的差别
    return cv2.bitwise_xor(img1, img2)


def calculate_lumitexel(img):
    # 直接转灰度图像
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return grayImage











