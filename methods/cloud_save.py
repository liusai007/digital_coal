"""
@Time : 2022/9/8 9:13 
@Author : ls
@DES: 
"""
import time
import numpy
import numpy as np
from pandas import DataFrame
from pyntcloud import PyntCloud
from scipy.spatial import Delaunay


def save_cloud(cloud: numpy.ndarray, file_path: str, as_ply: bool):
    mesh_name = 'cloud_data' + time.strftime('%d%H%M%S') + '.ply'
    full_name = file_path + '/' + mesh_name

    if cloud.size <= 10 * 3:
        return None
    else:
        pd_array = DataFrame(cloud, columns=["x", "y", "z"])
        x = cloud[:, 0]
        y = cloud[:, 1]
        tri = Delaunay(np.array([x, y]).T)
        mesh = DataFrame(tri.simplices, columns=['v1', 'v2', 'v3'])

        # diamond = PyntCloud(cloud=pd_array, mesh=mesh)
        diamond = PyntCloud(points=pd_array)
        diamond.mesh = mesh
        if as_ply:
            diamond.to_file(full_name, also_save=["mesh"], as_text=True)

        # str = "http://10.1.3.136:8001/ply/id/2022/09/14/qqq.ply"
        nginx_name = full_name.replace('/opt/python/coal_data', 'http://10.1.3.136:8001')
        return nginx_name
