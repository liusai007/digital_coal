"""
@Time : 2022/8/3 9:16 
@Author : ls
@DES: 
"""
from pandas import DataFrame
from pyntcloud import PyntCloud


def cloud_ndarray_sample(cloud_ndarray, n_x, n_y, n_z):
    # voxelgrid_centers,voxelgrid_centroids,
    # voxelgrid_nearest,voxelgrid_highest

    cloud_pdarray = DataFrame(cloud_ndarray[:, 0:3])
    cloud_pdarray.columns = ['x', 'y', 'z']
    cloud_pynt = PyntCloud(cloud_pdarray)
    voxelgrid_id = cloud_pynt.add_structure("voxelgrid", n_x=n_x, n_y=n_y, n_z=n_z)
    cloud_ndarray = cloud_pynt.get_sample("voxelgrid_highest",
                                          voxelgrid_id=voxelgrid_id, as_PyntCloud=False)
    cloud_ndarray = cloud_ndarray.values
    return cloud_ndarray
