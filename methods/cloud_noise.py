"""
@Time : 2022/8/17 8:47 
@Author : ls
@DES: 
"""
import numpy
from pandas import DataFrame
from pyntcloud import PyntCloud


def remove_noise(cloud: numpy.ndarray):
    cloud_array = DataFrame(cloud[:, 0:3])
    cloud_array.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    cloud_pynt = PyntCloud(points=cloud_array)  # 将points的数据 存到结构体中
    # cloud_pynt = PyntCloud(points=cloud_array, structures={'n_kdtrees': 15})  # 将points的数据 存到结构体中
    kdtree_id = cloud_pynt.add_structure(name="kdtree")
    # cloud_pynt = cloud_pynt.structures[kdtree_id]
    # cloud_pynt.get_filter("ROR", k=5, r=1.0, kdtree_id=kdtree_id, and_apply=True)
    cloud_pynt.get_filter(name="ROR", k=5, r=1.0, kdtree_id=kdtree_id, and_apply=True)
    # bool_array = cloud_pynt.get_filter(name="ROR", k=5, r=0.5, kdtree_id=kdtree_id, and_apply=False)
    new_cloud = cloud_pynt.xyz
    return new_cloud

