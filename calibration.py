import cv2
import numpy as np
import glob


objp = np.zeros((7 * 10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
objp = 2.6 * objp  # 打印棋盘格一格的边长为2.6cm
obj_points = []  # 存储3D点
img_points = []  # 存储2D点
images = glob.glob("/home/jing/PycharmProjects/heliostat_measure/cali_img/*.jpg")  # 黑白棋盘的图片路径

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        # cv2.drawChessboardCorners(img, (10, 7), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        # cv2.waitKey(1)
_, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)

# 内参数矩阵
Camera_intrinsic = {"mtx": mtx, "dist": dist, }