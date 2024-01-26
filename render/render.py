import cv2
import mitsuba as mt
import drjit as dj
import matplotlib.pyplot as plt
import numpy as np

import model.utils
from model.utils import *
np.random.seed(2333)

def render(scene_path, save_path):
    mt.set_variant("scalar_rgb")
    scene = mt.load_file(scene_path)
    image = mt.render(scene, spp=256)
    mt.util.write_bitmap(save_path, image)
render("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene.xml", "/home/jing/PycharmProjects/heliostat_measure/my_scene/test_new.jpg")
def get_random_alpha(low_broad, high_broad, num):
    y = np.logspace(low_broad, high_broad, num)
    return y
def get_random_pos(num, board_x, board_y):
    x = (board_x[1] - board_x[0]) * np.random.random((1, num)) + board_x[0]
    y = (board_y[1] - board_y[0]) * np.random.random((1, num)) + board_x[0]
    # 2 * np.random.random((3, 1)) - 1
    #
    # x = np.random.rand(board_x[0], board_x[1], num)
    # y = np.random.rand(board_y[0], board_y[1], num)
    return x, y
def render_sub_heiostat(alpha, pos):
    # 以pos为中心渲染一个长宽为20的微表面,返回pos点的像素值
    # 步骤1 生成pos为中心的一个微表面的obj
    # 步骤2 编辑xml文件,粗糙度为alpha
    # 步骤3 调取render函数进行渲染，返回pos的像素值
    scale = model.utils.get_scale()
def get_singleLight():
    scale = model.utils.get_scale()

    O = np.array([0.280551, 0.507936, 0.141849])
    X = np.array([-0.222432, 0.519973, 0.146457])
    Y = np.array([0.294847, 0.979378, 0.772472])
    rotation, transform = img2world_transform(O, X, Y)
    # length_pad = np.linalg.norm([F - G])
    # width_pad = np.linalg.norm([F - E])
    count = 0
    Z = 0
    for j in range(80):
        for i in range(128):
            temp = np.array([j * (20 / 2560) * scale * 22.62, i * (20 / 1600) * scale * 14.14,
                             (j + 1) * (20 / 2560) * scale * 22.62, (i + 1) * (20 / 1600) * scale * 14.14])
            print(temp)
            trans_vector1, trans_vector2, trans_vector3, trans_vector4 = (
                transformation(rotation, transform, np.array([temp[0], temp[1], Z])),
                transformation(rotation, transform, np.array([temp[2], temp[1], Z])),
                transformation(rotation, transform, np.array([temp[2], temp[-1], Z])),
                transformation(rotation, transform, np.array([temp[0], temp[-1], Z])))
            with open("/home/jing/PycharmProjects/heliostat_measure/my_scene/single_light/light_{}.obj".format(count),
                      'w') as thefile:
                thefile.write("v {0} {1} {2}\n".format(trans_vector1[0], trans_vector1[1], trans_vector1[2]))
                thefile.write("v {0} {1} {2}\n".format(trans_vector2[0], trans_vector2[1], trans_vector2[2]))
                thefile.write("v {0} {1} {2}\n".format(trans_vector3[0], trans_vector3[1], trans_vector3[2]))
                thefile.write("v {0} {1} {2}\n".format(trans_vector4[0], trans_vector4[1], trans_vector4[2]))
                thefile.write("f {0} {1} {2} {3}\n".format(4, 3, 2, 1))
            count += 1

def get_train_lumi(img_path, pos, calipath="camera_para_dict.npy"):
    # pos是世界坐标
    # 需要内外参进行转换
    img = cv2.imread(img_path)
    ret, matrix, dist, r_vecs, t_vecs = load_para(calipath)
    u, v = xyz2uv(matrix, r_vecs, t_vecs, pos)
    img_a = np.array(img)
    lumi = img_a[u, v]
    return lumi

import xml.etree.ElementTree as ET
def change_alpha_xml(path, alpha, index):
    # 1. 修改bsdf的alpha
    # 2. 修改obj heliostat的路径
    # --------------------------以上是100循环内
    # 3. 修改shape_light的路径
    # --------------------------以上是10240循环内
    tree = ET.parse(path)
    bsdf_node_u = get_node_by_keyvalue(find_nodes(tree, "bsdf/float"), {"name":"alpha_u"})
    bsdf_node_u[0].set("value", str(alpha))
    bsdf_node_v = get_node_by_keyvalue(find_nodes(tree, "bsdf/float"), {"name": "alpha_v"})
    bsdf_node_v[0].set("value", str(alpha))


    shape_node = get_node_by_keyvalue(find_nodes(tree, "shape"), {"id": "heliostat"})
    path_node = get_node_by_keyvalue(find_nodes(shape_node[0], "string"), {"name": "filename"})
    path_node[0].set("value", "/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/random_point_{}.obj".format(index))
    tree.write(path)

def change_light_xml(path, index):
    tree = ET.parse(path)
    shape_node = get_node_by_keyvalue(find_nodes(tree, "shape"), {"id": "light"})
    path_node = get_node_by_keyvalue(find_nodes(shape_node[0], "string"), {"name": "filename"})
    path_node[0].set("value", "/home/jing/PycharmProjects/heliostat_measure/my_scene/single_light/light_0.obj".format(index))

    tree.write(path)
    # bsdf_node = get_node_by_keyvalue(find_nodes(tree, "bsdf/float"), {"name":"alpha_u"})
    # bsdf_node[0].set("value", str(alpha))
# def change_light_xml(path, alpha):

# change_alpha_xml(path="/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_file_test.xml", alpha=10.0)



def main():
    scale = model.utils.get_scale()
    # 创建10240个单光源obj
    # 写完单个光源的obj文件 10240个
    # 范围内随机生成点p，在点p周围随机取一小块微定日镜


    alpha = get_random_alpha(0.0005, 0.01, 100)
    board_x = [0.25, 0.55]
    board_y = [0.3, 1.0]
    x_list, y_list = get_random_pos(100, board_x, board_y) # 随机选择中心点

    origin = np.array([-0.9099, 0.0278, 0.01316])
    x_axis = np.array([1.262660, 0.0117442, -0.013888])
    y_axis = np.array([-0.896279, 0.005022, 1.49966])
    width = np.linalg.norm(x_axis-origin)
    height = np.linalg.norm(y_axis-origin)

    Rotation, Translate = img2world_transform(origin, x_axis, y_axis)
    train_texel = []
    for i in range(100):
        # 生成100个渲染结果
        # 步骤：先随机取点，生成微定日镜，写入obj文件，obj文件+粗糙度
        # 循环singlelight里所有光源，每次使用一个光源照射obj文件(先obj文件循环，再xml文件光源修改循环)
        file1 = open("/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/random_point_{}.obj".format(i), 'w').close()
        # file2 = open("/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/random_xml.xml", 'w').close()

        alpha_temp, x, y = alpha[i], x_list[0, i], y_list[0, i]
        delta_x = width/20
        delta_y = height/20
        # helio_pos = np.array([x, y, 0])
        world_x, world_y, world_z = (transformation(Rotation, Translate, x),
                                     transformation(Rotation, Translate, y),
                                     transformation(Rotation, Translate, 0))
        world_pos = np.array([world_x, world_y, world_z])
        # temp_vector = np.array([x*width+delta_x, y*height+delta_y, 0]) # 镜面坐标系下的pos坐标
        left_up, right_up, right_bot, left_bot = (np.array([x*width-delta_x, y*height-delta_y, 0]),
                                                  np.array([x*width+delta_x, y*height-delta_y, 0]),
                                                  np.array([x*width+delta_x, y*height+delta_y, 0]),
                                                  np.array([x*width-delta_x, y*height+delta_y, 0]))
        left_up, right_up, right_bot, left_bot = (transformation(Rotation, Translate, left_up),
                                                  transformation(Rotation, Translate, right_up),
                                                  transformation(Rotation, Translate, right_bot),
                                                  transformation(Rotation, Translate, left_bot)) # 世界坐标系下的微表面4点坐标坐标

        with open("/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/random_point_{}.obj".format(i), 'w') as thefile:
            thefile.write("v {0} {1} {2}\n".format(left_up[0], left_up[1], left_up[2]))
            thefile.write("v {0} {1} {2}\n".format(right_up[0], right_up[1], right_up[2]))
            thefile.write("v {0} {1} {2}\n".format(right_bot[0], right_bot[1], right_bot[2]))
            thefile.write("v {0} {1} {2}\n".format(left_bot[0], left_bot[1], left_bot[2]))
            thefile.write("f {0} {1} {2} {3}\n".format(1, 2, 3, 4))
            thefile.flush()
        change_alpha_xml(path="/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_file_test.xml", alpha=alpha_temp, index=i)
        train_texel = []
        for j in range(10240):
            change_light_xml(path="/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_file_test.xml", index=j)
            # obj文件生成结束, xml修改结束，渲染
            render(scene_path="/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_file_test.xml",
                   save_path="/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_render/temp_render.png")
            lumitexel = get_train_lumi("/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_render/temp_render.png", world_pos)
            train_texel.append(lumitexel)
    # 100*10240的生成结束，存储起来
    np.save("/home/jing/PycharmProjects/heliostat_measure/my_scene/train_scene/train_file.npy", train_texel)

# main()

# render('', '')