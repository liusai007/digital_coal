"""
@Time : 2022/9/9 10:50 
@Author : ls
@DES: 
"""
import numpy
from typing import List
from radars.config import RADARS_MATRIX
from models.custom_class import CoalRadar


def cloud_transform(cloud: numpy.ndarray, radar: CoalRadar):
    ip = radar.ip
    matrix = RADARS_MATRIX.get(ip)
    new_cloud = numpy.dot(matrix, cloud.T)

    matrix_array = new_cloud.T
    new_cloud = numpy.array(matrix_array)
    new_cloud = new_cloud[:, :3]
    return new_cloud


def radars_cloud_transform(radars: List[CoalRadar]):
    ndarray_list = []
    for radar in radars:
        new_array = radar_cloud_transform(radar)
        ndarray_list.append(new_array)

    cloud = numpy.concatenate(ndarray_list, axis=0)
    return cloud


def radar_cloud_transform(radar: CoalRadar):
    bytes_data = radar.bytes_buffer
    cloud = numpy.frombuffer(bytes_data, dtype=numpy.int16).reshape(-1, 3)
    div = numpy.array([100, 100, 100])
    cloud = numpy.divide(cloud, div)
    cloud = numpy.insert(cloud, 3, values=1.0, axis=1)
    # cloud.astype(numpy.float16)

    ip = radar.ip
    matrix = RADARS_MATRIX.get(ip)
    matrix = numpy.array(matrix)
    new_cloud = numpy.dot(matrix, cloud.T)

    new_cloud = new_cloud.T
    new_cloud = new_cloud[:, :3]
    return new_cloud

