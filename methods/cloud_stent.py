"""
@Time : 2022/8/24 15:18 
@Author : ls
@DES: 
"""
import numpy
from typing import List
from models.custom_stent import Stent


def remove_stents(cloud: numpy.ndarray, stent_list: List[Stent]):
    for stent in stent_list:
        x_range = (cloud[:, 0] < stent.min_x) | (cloud[:, 0] > stent.max_x)
        y_range = (cloud[:, 1] < stent.min_y) | (cloud[:, 1] > stent.max_y)
        cloud = cloud[numpy.where(x_range | y_range)]

    return cloud
