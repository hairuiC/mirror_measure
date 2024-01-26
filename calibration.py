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
_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
#     return img
#
#
# def calibration_photo(photo_path):
#     # 设置要标定的角点个数
#     x_nums = 10  # x方向上的角点个数
#     y_nums = 7
#     # 设置(生成)标定图在世界坐标中的坐标
#     world_point = np.zeros((x_nums * y_nums, 3), np.float32)  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
#     world_point[:, :2] = 15 * np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)  # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
#     print('world point:', world_point)
#     # .T矩阵的转置
#     # reshape()重新规划矩阵，但不改变矩阵元素
#     # 设置世界坐标的坐标
#     axis = 15 * np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
#     # 设置角点查找限制
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     image = cv2.imread(photo_path)
#
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # 查找角点
#     ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), )
#     print(ok)
#     if ok:
#         # 获取更精确的角点位置
#         exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#
#         # 获取外参
#         _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
#         # 获得的旋转矩阵是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换Rodrigues，
#
#         rotation_m, _ = cv2.Rodrigues(rvec)  # 罗德里格斯变换
#         # print(rotation_m)
#         # print('旋转矩阵是：\n', rvec)
#         # print('平移矩阵是:\n', tvec)
#         rotation_t = np.hstack([rotation_m, tvec])
#         rotation_t_Homogeneous_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])
#         print(rotation_t_Homogeneous_matrix)
#         imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
#         # 可视化角点
#         img = draw(image, corners, imgpts)
#         cv2.imshow('img', img)
#         return rotation_t_Homogeneous_matrix  # 返回旋转矩阵和平移矩阵组成的其次矩阵


# 内参数矩阵
Camera_intrinsic = {"mtx": mtx, "dist": dist, "外参旋转": rvecs, "外参平移":tvecs}
# photo_path = "/home/jing/PycharmProjects/heliostat_measure/cali_img/cali20.jpg" # 标定图像保存路径
# calibration_photo(photo_path)
# cv2.waitKey()
# cv2.destroyAllWindows()