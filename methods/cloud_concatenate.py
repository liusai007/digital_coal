import math
import time
import numpy as np
from typing import List
from models.custom_class import CoalRadar
from pydantic import BaseModel
import scipy.linalg as linalg


class cloud_path():
    def __init__(self, id: int, path: str):
        self.id = id
        self.path = path


def cloud_concatenate(cloud_paths, radars: List[CoalRadar], save_path):
    # cloud_paths: ，包含 雷达id和对应的文件路径
    '''
    :param cloud_paths: 雷达扫描后返回的 文件列表
    :param radars: 雷达对象
    :param save_path: 保存拼接后的点云的路径
    :return: 拼接后的 ndarray 对象
    '''
    clouds_list = []
    for c_path in cloud_paths:
        for radar in radars:
            if c_path[0] == radar.id:
                new_cloud = cloud_convert(c_path[1], radar)
                clouds_list.append(new_cloud)

    combined_cloud_ndarray = np.concatenate(clouds_list, axis=0)
    np.savetxt(save_path, combined_cloud_ndarray, fmt='%d')

    return combined_cloud_ndarray


def cloud_convert(cloud_path, radar: CoalRadar, save_path=None):
    # 点云路径，为 txt 格式
    axis_x = [1, 0, 0]
    axis_y = [0, 1, 0]
    axis_z = [0, 0, 1]
    # 旋转角度
    x_radian = radar.rotateX * math.pi / 180
    y_radian = radar.rotateY * math.pi / 180
    z_radian = radar.rotateZ * math.pi / 180
    # 旋转矩阵常量
    x_matrix = linalg.expm(np.cross(np.eye(3), axis_x / linalg.norm(axis_x) * x_radian))
    y_matrix = linalg.expm(np.cross(np.eye(3), axis_y / linalg.norm(axis_y) * y_radian))
    z_matrix = linalg.expm(np.cross(np.eye(3), axis_z / linalg.norm(axis_z) * z_radian))

    cloud_array = np.genfromtxt(cloud_path)
    # 点云旋转操作
    matrix = np.dot(x_matrix, np.dot(y_matrix, z_matrix))
    rotate_cloud_array = np.dot(cloud_array, matrix)

    # 点云平移操作
    shift_xyz = np.array([[radar.shiftX, radar.shiftY, radar.shiftZ]])
    new_cloud_array = rotate_cloud_array + shift_xyz

    # np.savetxt(save_path, new_cloud_array, fmt='%d')
    return new_cloud_array
