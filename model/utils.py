import torch
import numpy as np
import math
import torch.nn.init
import glob
import cv2
import os.path
import os
import xml
from xml.etree.ElementTree import ElementTree,Element
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.etree.ElementTree import ElementTree,Element

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



def inverse_color(grey_img):
    # 反色函数
    return cv2.bitwise_not(grey_img)
    # return Inverse_frame_gray

def img_xor(img1, img2):
    # 图像异或，主要用来展示不同进光量摄像头拍摄图像的差别
    return cv2.bitwise_xor(img1, img2)


def calculate_lumitexel(pos, dirpath):
    # lumitexel是某个点来说的
    X, Y = pos[0], pos[1]
    images = glob.glob(dirpath + os.sep + "*.jpg")
    lumitexel_pos = []
    for img in images:
        lumitexel_pos.append(img[X, Y])
    return lumitexel_pos




    # return grayImage

class criterion():
    def __init__(self, input, GT, weight, epsilon, labda):
        self.input = input
        self.GT = GT
        self.weight = weight
        self.epsilon = epsilon
        self.labda = labda
    def DAE_Loss(self):
        loss_item1 = 0
        loss_item2 = 0
        # self.epsilon = 0.005
        # labda = 0.3
        for i in range(len(self.input)):
            loss_item1 += math.sqrt(math.log2(1 + self.input[i]) - math.log2(1 + self.GT[i]))
            loss_item2 += math.tanh((self.weight[i] - (1 - self.epsilon)) / self.epsilon) + math.tanh((-self.weight[i] + self.epsilon) / self.epsilon) + 2
        return loss_item1 + self.labda * loss_item2

def get_lightingPattern(model):
    return model.encoder[0].weight, model.encoder[0].weight.grad

def no_learningConv2d(kernel_size, stride, input):
    kernel = np.ones((kernel_size, kernel_size))
    # i = 0;
    # j = 0;
    # print(i)
    H,W = input.shape
    # print(len(input[0]))
    # print(len(input[1]))
    # print(H/stride, W/stride)
    res = np.ones((int(H/stride), int(W/stride)))
    for i in range(0, int(H), stride):
        for j in range(0, int(W), stride):
            temp_value = (sum(sum(np.multiply(kernel, input[i:i+stride, j:j+stride]))))/400
            print(((i+stride)/stride)-1, ((j + stride)/stride)-1)
            res[int(((i+stride)/stride)-1)][int(((j + stride)/stride)-1)] = temp_value
    return res



def img2world_transform(origin, x, y):
    #坐标系转换函数，输入两个坐标系的原点、轴，返回变换矩阵
    axis_x = np.array((x - origin) / np.linalg.norm(x - origin))
    axis_y = np.array((y - origin) / np.linalg.norm(y - origin))
    axis_z = np.array(np.cross(axis_x, axis_y) / np.linalg.norm(np.cross(axis_x, axis_y)))
    Rotation = np.concatenate((axis_x.reshape(-1,1), axis_y.reshape(-1,1), axis_z.reshape(-1,1)), axis=1)
    Translate = origin
    return Rotation, Translate

def transformation(rotate, transfom, mat_B):
    return (np.dot(rotate, mat_B) + transfom)


def write_obj(lighting_pattern, origin, x, y, scale, index):
    # scale = calculate_scale()
    render_mat = []
    # temp = []
    rotation, transform = img2world_transform(origin, x, y)
    X, Y = np.nonzero(lighting_pattern)
    # 22.62, 14.14是实际尺寸，单位是cm
    for i in range(len(X)):
        render_mat.append([X[i]*(20/2560)*scale*22.62, Y[i]*(20/1600)*scale*14.14, (X[i]+1)*(20/2560)*scale*22.62, (Y[i]+1)*(20/1600)*scale*14.14])
    with open('/home/jing/PycharmProjects/heliostat_measure/my_scene/emitter/test_{}.obj'.format(index), 'w') as thefile:
    # thefile = open('test.obj', 'w')
        faces = []
        for j, items in enumerate(render_mat):
            X0, Y0, X1, Y1 = items[0], items[1], items[2], items[3]
            Z = 0
            trans_vector1, trans_vector2, trans_vector3, trans_vector4 = (transformation(rotation, transform, np.array([X0, Y0, Z])),
                                                                          transformation(rotation, transform, np.array([X1, Y0, Z])),
                                                                          transformation(rotation, transform, np.array([X1, Y1, Z])),
                                                                          transformation(rotation, transform, np.array([X0, Y1, Z])))
            # new1 = transform(rotation, transform, np.array([X0, Y0, Z]))
            thefile.write("v {0} {1} {2}\n".format(trans_vector1[0], trans_vector1[1], trans_vector1[2]))
            thefile.write("v {0} {1} {2}\n".format(trans_vector2[0], trans_vector2[1], trans_vector2[2]))
            thefile.write("v {0} {1} {2}\n".format(trans_vector3[0], trans_vector3[1], trans_vector3[2]))
            thefile.write("v {0} {1} {2}\n".format(trans_vector4[0], trans_vector4[1], trans_vector4[2]))
            faces.append("f {0} {1} {2} {3}\n".format(j*4+4, j*4+3, j*4+2, j*4+1))
        for face in faces:
            thefile.write(face)
    thefile.close()

    edit_xml("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back[0]ml",
             index,
             "/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back[0]ml",
             '/home/jing/PycharmProjects/heliostat_measure/my_scene/emitter/test_{}.obj'.format(index))
    to_pretty_xml("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene_back[0]ml")

def get_panel(p1, p2, p3):
    p1 = np.array([0.280551, 0.507936, 0.141849])
    p2 = np.array([-0.222432, 0.519973, 0.146457])
    p3 = np.array([0.294847, 0.979378, 0.772472])

    a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
    b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
    c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    d = (0 - (a * p1[0] + b * p1[1] + c * p1[2]))
    return a, b, c, d




def calculate_scale(real_cm, obj_cm):
    scale_objPerCM = obj_cm / real_cm
    return scale_objPerCM

def get_training_data(N, dir):
    # 随机选N个点组成训练数据
    training_data = []

    for i in range(N):
        lumitexel_pos = []
        [X, Y] = np.random.uniform(0, 1, 2)
        X = int(X * 2050)
        Y = int(Y * 3223)
        lumitexel_pos = calculate_lumitexel([X, Y], dir)
        training_data.append(lumitexel_pos)
    return training_data

def get_scale():
    B = np.array([1.262660, 0.0117442, -0.013888])
    C = np.array([1.277871, -0.002134, 1.471240])
    vector_BC = C - B
    obj_length = np.linalg.norm(vector_BC)
    return obj_length/40

# -------------------------------------------------
#
# 以下为xml相关
#
# -------------------------------------------------

def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def if_match(node, kv_map):
    '''''判断某个节点是否包含所有传入参数属性
       node: 节点
       kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True

# ----------------search -----------------
def find_nodes(tree, path):
    '''''查找某个路径匹配的所有节点
       tree: xml树
       path: 节点路径'''
    return tree.findall(path)

def get_node_by_keyvalue(nodelist, kv_map):
    '''''根据属性及属性值定位符合的节点，返回节点
       nodelist: 节点列表
       kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes

# ---------------change ----------------------
def change_node_properties(nodelist, kv_map, is_delete=False):
    '''修改/增加 /删除 节点的属性及属性值
       nodelist: 节点列表
       kv_map:属性及属性值map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete:
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))

def change_node_text(nodelist, text, is_add=False, is_delete=False):
    '''''改变/增加/删除一个节点的文本
       nodelist:节点列表
       text : 更新后的文本'''
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text

def create_node(tag, property_map, content=None):
    '''新造一个节点
       tag:节点标签
       property_map:属性及属性值map
       content: 节点闭合标签里的文本内容
       return 新节点'''
    element = Element(tag, property_map)
    if content != None:
        element.text = content
    return element

def add_child_node(nodelist, element):
    '''''给一个节点添加子节点
       nodelist: 节点列表
       element: 子节点'''
    for node in nodelist:
        node.append(element)


def del_node_by_tagkeyvalue(nodelist, tag, kv_map):
    '''''同过属性及属性值定位一个节点，并删除之
       nodelist: 父节点列表
       tag:子节点标签
       kv_map: 属性及属性值列表'''
    for parent_node in nodelist:
        children = parent_node.getchildren()
        for child in children:
            if child.tag == tag and if_match(child, kv_map):
                parent_node.remove(child)

def edit_xml(in_path, index, out_path, obj_path):

    tree = read_xml(in_path)
    root = tree.getroot()
    element = Element("shape", {"type": "obj", "id": "light{}".format(index+1)})
    root.append(element)
    tree.write(out_path, encoding="utf-8",
               xml_declaration=True)

    tree = read_xml(out_path)
    # root = tree.getroot()
    # print(root)
    nodes = find_nodes(tree, "shape")

    result_nodes = get_node_by_keyvalue(nodes, {"id": "light{}".format(index+1)})
    a = create_node("string", {"name": "filename", "value": obj_path})
    print(result_nodes)
    add_child_node(result_nodes, a)

    b = create_node("emitter", {"type": "area"})
    add_child_node(result_nodes, b)
    tree.write(out_path, encoding="utf-8",
               xml_declaration=True)

    tree = read_xml(out_path)
    nodes = find_nodes(tree, "shape")
    node_4 = get_node_by_keyvalue(nodes, {"id": "light{}".format(index+1)})
    children1 = node_4[0].iter()
    for child in children1:
        if child.tag == "emitter" and if_match(child, {"type": "area"}):
            # print(child)
            c = create_node("rgb", {"name": "radiance", "value": "{}".format(255*(index+1)*0.2)})
            add_child_node([child], c)
            break
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
def to_pretty_xml(path):
    file = open(path, 'r')
    xml_string = file.read()
    file.close()

    parsed_xml = xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = parsed_xml.toprettyxml()

    file = open(path, 'w')
    file.write(pretty_xml_as_string)
    file.close()
def del_empty(in_path, out_path):
    file_in = open(in_path, 'r')
    file_out = open(out_path, 'w')
    for line in file_in.readlines():
        if line.isspace():
            continue
        file_out.write(line)
    file_in.close()
    file_out.close()
def copy_file(in_path, out_path):
    file_in = open(in_path, 'r')
    file_out = open(out_path, 'w')
    for line in file_in.readlines():
        if line.isspace():
            continue
        file_out.write(line)
    file_in.close()
    file_out.close()
# -------------------------------------------------------------
#
# 以下为标定相关
#
# -------------------------------------------------------------
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
    obj_pts = np.array(obj_pts)
    img_pts = np.array(img_pts)
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (h, w), None, None)
    print(mtx)

    # _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)
    r_mtx = cv2.Rodrigues(rvecs[0])[0]
    t_mtx = tvecs[0]
    print(r_mtx)
    print(tvecs[0])
    # rt = np.array([[r_mtx[0][0], r_mtx[0][1], tvec[0]],
    #                [],
    #                []])


    dic_distortion['ret'] = ret
    dic_distortion['mtx'] = mtx
    dic_distortion['dist'] = dist
    dic_distortion['r_mtx'] = r_mtx
    dic_distortion['t_mtx'] = t_mtx

    save_para(dic_distortion)

    return dic_distortion

# def cal_rt(mtx, dist, chess_path):
#     obj_p = np.zeros((9 * 6, 3), np.float32)
#     obj_p[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
#     objp = 2.6 * obj_p
#
#     obj_points = []
#     img_points = []
#
#     obj_points = objp
#
#     get_path = chess_path + "/*.jpg"
#     images = glob.glob(get_path)
#     for img in images:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.drawChessboardCorners(gray, (9, 6), None)
#         if ret:
#             img_points = np.array(corners)
#             cv2.drawChessboardCorners(gray, (9, 6), corners, ret)
#             _, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
#             r_mtx = cv2.Rodrigues(rvec)[0]
#
#             rt = np.array([[r_mtx[0][0], r_mtx[0][1], tvec[0]],
#                            [r_mtx[1][0], r_mtx[1][1], tvec[1]],
#                            [r_mtx[2][0], r_mtx[2][1], tvec[2]]], dtype=np.float)
#
#             rt_i = np.linalg.inv(rt)
#             pi_i = np.linalg.inv(mtx)






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
    matrix = para["mtx"]
    dist = para["dist"]
    r_mtx = para["r_mtx"]
    t_mtx = para["t_mtx"]

    return ret, matrix, dist, r_mtx, t_mtx

def xyz2uv(mtx, r_mtx, t_mtx, world_pos):
    focal_x = mtx[0][0]
    focal_y = mtx[1][1]

    offset_x = mtx[0][2]
    offset_y = mtx[1][2]
    # world_p = np.array([x, y, z, 0])
    camera_p = np.dot(world_pos - t_mtx, r_mtx)
    x, y, z = camera_p[0], camera_p[1], camera_p[2]

    new_x = -x * focal_x/z + offset_x
    new_y = y * focal_y/z+offset_y

    screen_p = np.array([new_x, new_y])

    return screen_p

# def xyz2uv(matrix, r_vecs, t_vecs, w_pos):
#     last_row = np.array([0, 0, 0, 1])
#     out_mat = np.hstack((r_vecs, t_vecs))
#     out_mat = np.vstack((out_mat, last_row))
# 
#     w_pos = np.hstack((w_pos, [1]))
#     camera_coor = w_pos * w_pos
#     Z_c = float(camera_coor[2])
#     img_coor = matrix*(camera_coor /Z_c)
# 
#     u,v = img_coor[0], img_coor[1]
#     return u, v
