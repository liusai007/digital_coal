import numpy
import math
import numpy as np
from typing import List
import scipy.linalg as linalg
from pydantic import BaseModel
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud
from pyntcloud.io import to_open3d


class CoalRadar(BaseModel):
    axisX: float = 0.0
    axisY: float = 0.0
    id: int = 1
    ip: str = None
    name: str = '1号雷达'
    port: int = 7999
    rotateX: float = 0.0
    rotateY: float = 0.0
    rotateZ: float = 0.0
    shiftX: float = 0.0
    shiftY: float = 0.0
    shiftZ: float = 0.0
    filename: str = 'radar_200.txt'


radar_200 = CoalRadar()
radar_200.axisX = 0.0
radar_200.axisY = 0.0
radar_200.rotateX = 0.0
radar_200.rotateY = 0.0
radar_200.rotateZ = 0.0
radar_200.shiftX = 0.0
radar_200.shiftY = 0.0
radar_200.shiftZ = 0.0
radar_200.filename = 'radar_200.txt'

radar_201 = CoalRadar()
radar_201.axisX = 0.0
radar_201.axisY = 0.0
radar_201.rotateX = -14.5
radar_201.rotateY = -5.8
radar_201.rotateZ = -20.0
radar_201.shiftX = 75.9
radar_201.shiftY = -95.8
radar_201.shiftZ = -17.0
radar_201.filename = 'radar_201.txt'


def euler_rotate_and_horizontal_offset(radar_list: List[CoalRadar]):
    ndarray_list: List[numpy.ndarray] = []

    axis_x = [1, 0, 0]
    axis_y = [0, 1, 0]
    axis_z = [0, 0, 1]

    for radar_1 in radar_list:
        cloud_ndarray_1 = np.genfromtxt(radar_1.filename)
        # cloud_ndarray_2 = np.genfromtxt(radar_2.filename)

        # 旋转角度
        x_1_radian = radar_1.rotateX * math.pi / 180
        y_1_radian = radar_1.rotateY * math.pi / 180
        z_1_radian = radar_1.rotateZ * math.pi / 180
        # 旋转矩阵常量
        x_1_matrix = linalg.expm(np.cross(np.eye(3), axis_x / linalg.norm(axis_x) * x_1_radian))
        y_1_matrix = linalg.expm(np.cross(np.eye(3), axis_y / linalg.norm(axis_y) * y_1_radian))
        z_1_matrix = linalg.expm(np.cross(np.eye(3), axis_z / linalg.norm(axis_z) * z_1_radian))

        # 点云旋转操作
        matrix_1 = np.dot(x_1_matrix, np.dot(y_1_matrix, z_1_matrix))
        new_cloud_array_1 = np.dot(cloud_ndarray_1, matrix_1)

        # 点云平移操作
        shift_xyz_1 = np.array([[radar_1.shiftX, radar_1.shiftY, radar_1.shiftZ]])
        new_cloud_array_1 = new_cloud_array_1 + shift_xyz_1
        ndarray_list.append(new_cloud_array_1)

    combined_ndarray = numpy.concatenate(ndarray_list, axis=0)

    np.savetxt('combined_cloud.txt', X=combined_ndarray, delimiter=' ', fmt='%.2f')
    cloud_array = DataFrame(combined_ndarray[:, 0:3])
    cloud_array.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    cloud_pynt = PyntCloud(cloud_array)  # 将points的数据 存到结构体中

    o3d_cloud = to_open3d(cloud_pynt, mesh=True)
    o3d_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([o3d_cloud])


radar_list = [radar_200, radar_201]
euler_rotate_and_horizontal_offset(radar_list=radar_list)
