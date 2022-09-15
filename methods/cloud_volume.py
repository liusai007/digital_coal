import vtk
import numpy
import numpy as np
from pandas import DataFrame
from datetime import datetime
from scipy.spatial import Delaunay
from pyntcloud import PyntCloud


async def heap_vom_and_maxheight(cloud_ndarray: numpy.ndarray, minio_path: str = None):
    # cloud_ndarray = cloud_ndarray_sample(cloud_ndarray, n_x=200, n_y=200, n_z=100)
    # 三角剖分开始
    if cloud_ndarray.size < 3 * 20:
        return {'maxHeight': 0, 'volume': 0}

    u = cloud_ndarray[:, 0]
    v = cloud_ndarray[:, 1]
    z = cloud_ndarray[:, 2]
    x = u
    y = v

    maxHeight = max(z) - min(z)

    ply_name = minio_path.replace('.txt', '.ply')
    tri = Delaunay(np.array([u, v]).T)
    f2 = open(ply_name, 'w', encoding='utf-8')
    f2.write('ply\n')
    f2.write('format ascii 1号雷达.0\n')
    f2.write('comment Created by CloudCompare v2.11.3 (Anoia)\n')
    f2.write('comment Created 2021/12/26 19:23\n')
    f2.write('obj_info Generated by CloudCompare!\n')
    f2.write('element vertex ')
    f2.write(str(x.size) + '\n')
    f2.write('property float x\n')
    f2.write('property float y\n')
    f2.write('property float z\n')
    f2.write('element face ')
    f2.write(str(tri.simplices.shape[0]) + '\n')
    f2.write('property list uchar int vertex_indices\n')
    f2.write('end_header\n')
    for i in range(x.shape[0]):
        f2.write('%.2f %.2f %.2f\n' % (x[i], y[i], z[i]))
    for vert in tri.simplices:
        f2.write('3 %d %d %d\n' % (vert[0], vert[1], vert[2]))
    f2.close()

    # 凸面计算体积
    diamond = PyntCloud.from_file(ply_name)
    convex_hull_id = diamond.add_structure("convex_hull")
    convex_hull = diamond.structures[convex_hull_id]
    diamond.mesh = convex_hull.get_mesh()
    # diamond.to_file("bunny_hull.ply", also_save=["mesh"])
    volume = convex_hull.volume
    # 三角剖分结束
    response = {'maxHeight': maxHeight, 'volume': volume}
    return response


async def new_heap_vom_and_maxheight(cloud_ndarray: numpy.ndarray, minio_path: str = None):
    if cloud_ndarray.size < 3 * 20:
        return {'maxHeight': 0, 'volume': 0}

    pd_array = DataFrame(cloud_ndarray)
    pd_array.columns = ['x', 'y', 'z']
    u = cloud_ndarray[:, 0]
    v = cloud_ndarray[:, 1]
    z = cloud_ndarray[:, 2]

    maxHeight = max(z) - min(z)

    tri = Delaunay(np.array([u, v]).T)
    # nn = np.array([u, v]).T
    # mesh = DataFrame(tri.simplices)
    mesh = DataFrame(tri.vertices)
    mesh.columns = ['v1', 'v2', 'v3']

    diamond = PyntCloud(points=pd_array, mesh=mesh)
    convex_hull_id = diamond.add_structure("convex_hull")
    convex_hull = diamond.structures[convex_hull_id]
    diamond.mesh = convex_hull.get_mesh()
    # diamond.to_file("bunny_hull.ply", also_save=["mesh"])
    volume = convex_hull.volume

    response = {'maxHeight': maxHeight, 'volume': volume}
    return response


async def ply_heap_vom_and_height(cloud_ndarray: numpy.ndarray):
    if cloud_ndarray.size < 3 * 20:
        return {'maxHeight': 0, 'volume': 0}

    pd_array = DataFrame(cloud_ndarray, columns=['x', 'y', 'z'])
    u = cloud_ndarray[:, 0]
    v = cloud_ndarray[:, 1]
    z = cloud_ndarray[:, 2]

    max_height = max(z) - min(z)

    tri = Delaunay(np.array([u, v]).T)
    mesh = DataFrame(tri.simplices, columns=['v1', 'v2', 'v3'])

    diamond = PyntCloud(points=pd_array, mesh=mesh)
    diamond.to_file("dd.ply", also_save=["mesh"])

    # ply体积测量
    vtkReader = vtk.vtkPLYReader()
    vtkReader.SetFileName("dd.ply")
    vtkReader.Update()
    polydata = vtkReader.GetOutput()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)

    volume = mass.GetVolume()

    response = {'maxHeight': max_height, 'volume': volume}
    return response
