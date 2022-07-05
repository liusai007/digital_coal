"""
@Time : 2022/6/16 11:48
@Author : ls
@DES:
"""
import os
import math
import time
import numpy
import minio
import platform
import numpy as np
from datetime import datetime
from pandas import DataFrame
from pyntcloud import PyntCloud
import scipy.linalg as linalg
from config import settings
from fastapi import APIRouter
from typing import List
from queue import Queue
from threading import Thread
from scipy.spatial import Delaunay
from multiprocessing import Process
from multiprocessing import JoinableQueue
from models.custom_class import CoalYard, CoalRadar, InventoryCoalResult
from core.Response import success, fail
from pydantic import BaseModel
from methods.bytes_to_txt import write_bytes_to_txt
# from methods.convert_bin_by_radar import convert_bin_by_radar
from ctypes import *

router = APIRouter()
queue = Queue()
# queue = JoinableQueue()

gCallbackFuncList = []  # 定义全局列表接收回调，防止系统回收
dllPath = settings.STATIC_DIR + '/sdk/CDPSIMCLIENT.dll'
soPath = settings.STATIC_DIR + '/sdk/cdpsimclient-linux64.so'

if platform.system() == 'Windows':
    from ctypes import windll

    dll = windll.LoadLibrary(dllPath)
else:
    from ctypes import cdll

    dll = cdll.LoadLibrary(soPath)


@router.post("/radar_test", summary="雷达测试")
def radar_start(CoalYard: CoalYard):
    resList = list()  # 设置一个空字典，接收煤堆对象

    global bin_cloud_list, all_cloud_list
    bin_cloud_list = list()
    all_cloud_list = list()
    global runmode, speed, AngleSceneScan, create_time, conn_radar, stop_radar
    runmode = 0
    speed = 10
    conn_radar = list()
    stop_radar = list()
    AngleSceneScan = 360
    create_time = int(time.strftime('%m%d%H%M%S'))
    init_status = dll.NET_SDK_SIMCLT_Init()
    print("初始化结果:", init_status)

    # 设置值回调函数
    radars = CoalYard.coalRadarList
    CALLBACK = WINFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    # CALLBACK = CFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    callBackFunc = CALLBACK(_callback)
    gCallbackFuncList.append(callBackFunc)
    dll.NET_SDK_SIMCLT_Set_Callback(callBackFunc, create_time)

    try:
        for radar in radars:
            cid = radar.id
            ip = bytes(radar.ip, encoding='utf-8')
            port = radar.port
            res = dll.NET_SDK_SIMCLT_StartConnect(cid, ip, port, 120)
            print('连接状态:', res, 'cid ==', cid)

        time.sleep(2.5)
        while True:
            if len(stop_radar) == len(conn_radar):
            # if len(bin_cloud_list) == len(conn_radar):  # 判断生成的数据文件 与 雷达数据 是否匹配
                dll.NET_SDK_SIMCLT_Destory()
                print("===========销毁sdk============")
                break
    except:
        # 程序报错，就销毁 sdk
        dll.NET_SDK_SIMCLT_Destory()

    startTime = datetime.now()
    for bin_cloud in bin_cloud_list:
        for radar in radars:
            if bin_cloud[0] == radar.id:
                bin_filename = bin_cloud[1]
                new_thread = Thread(target=convert_bin_by_radar, args=(bin_filename, radar,))
                new_thread.start()
                new_thread.join()

    print("我在等待转换，我可不能出现==============")
    combined_cloud_path = settings.CLOUD_COMBINED_PATH
    sampled_cloud_path = settings.CLOUD_SAMPLED_PATH
    if not os.path.exists(sampled_cloud_path):
        os.makedirs(sampled_cloud_path)

    # 合并后的点云进行抽稀去重操作
    print('====================')
    combined_filename = combined_cloud_path + "/combined_cloud_" + str(create_time) + ".txt"
    combined_cloud_ndarray = np.concatenate(all_cloud_list, axis=0)
    np.savetxt(combined_filename, combined_cloud_ndarray, fmt='%.2f', delimiter=' ')
    # combined_cloud_ndarray = np.concatenate(all_cloud_list, axis=0).astype(np.int16)
    combined_cloud_pdarray = DataFrame(combined_cloud_ndarray[:, 0:3])
    combined_cloud_pdarray.columns = ['x', 'y', 'z']
    combined_cloud_pynt = PyntCloud(combined_cloud_pdarray)
    voxelgrid_id = combined_cloud_pynt.add_structure("voxelgrid", n_x=100, n_y=100, n_z=50)
    # 下面可能会报内存爆满错误 ！！！
    sampled_cloud_pdarray = combined_cloud_pynt.get_sample("voxelgrid_centroids", voxelgrid_id=voxelgrid_id,
                                                           as_PyntCloud=False)
    sampled_filename = sampled_cloud_path + "/sampled_cloud_" + str(create_time) + ".txt"
    sampled_cloud_pdarray.to_csv(sampled_filename, index=False, header=False, sep=' ', float_format='%.6f')
    # new_combined_ndarray = sampled_cloud_pdarray.values  # DataFrame 转 ndarray
    # np.savetxt(combined_filename, new_combined_ndarray, fmt='%.6f', delimiter=' ')
    endTime = datetime.now()
    print('点云旋转平移去重稀释完成，耗时==', endTime - startTime)

    # 判断yard_name 文件夹是否存在，不存在创建
    coal_yard_path = settings.DATA_PATH + '/' + CoalYard.coalYardName
    if not os.path.exists(coal_yard_path):
        os.makedirs(coal_yard_path)
    for coal_heap in CoalYard.coalHeapList:
        res = InventoryCoalResult()
        res.coalHeapId = coal_heap.coalHeapId
        res.coalHeapName = coal_heap.coalHeapName
        res.density = coal_heap.density
        res.mesId = coal_heap.mesId
        print('coal_heap ==', coal_heap)
        minio_name = 'coalHeap' + str(coal_heap.coalHeapId) + '_' + str(create_time) + '.txt'
        minio_path = coal_yard_path + '/' + minio_name
        vom_and_maxhei_and_minio = heap_volume_and_maxheight(coal_heap, sampled_cloud_pdarray,
                                                             minio_path=minio_path, minio_name=minio_name)
        # res.cloudInfo = put_cloud(minio_path, minio_name)
        res.cloudInfo = vom_and_maxhei_and_minio['minio_url']
        res.volume = vom_and_maxhei_and_minio['volume']
        res.maxHeight = vom_and_maxhei_and_minio['maxHeight']

        resList.append(res)

    last_time = datetime.now()
    print("============程序总耗时 =========== ", last_time - startTime)
    return success(msg='盘煤成功', data=resList)


def _callback(cid: c_uint, datalen: c_int, data, createe_time):
    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    filepath = settings.DATA_PATH + '/' + str(cid)
    if not os.path.exists(path=filepath):
        os.makedirs(filepath)
    filename = filepath + '/cloudData_' + str(create_time) + '.bin'
    file = open(filename, mode='ab+')

    if code == 3534:
        dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, runmode, 64, 0, 360)
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, speed, 0, AngleSceneScan)
        conn_radar.append(cid)
    elif code == 3535:
        print("连接失败")
        file.close()
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_StopConnectCid(cid)
        # bin_cloud_list.append([cid, filename])
        # stop_radar.append(cid)
    elif code == 51108:
        print("运行模式设置成功")
    elif code == 118:
        points_data = data[54:datalen]
        file.write(points_data)

        lastLineFlag = data[44]
        if lastLineFlag == b'\x80':
            file.close()
            stop_radar.append(cid)
            dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
            dll.NET_SDK_SIMCLT_StopConnectCid(cid)
            bin_cloud_list.append([cid, filename])
            # print("bin_cloud_list ==", bin_cloud_list)
    else:
        pass
    return


def convert_bin_by_radar(bin_filename, radar):
    bin_ndarray = np.fromfile(bin_filename, dtype=np.int16)
    new_ndarray = bin_ndarray.reshape(-1, 3)
    div = np.array([100, 100, 100])
    new_ndarray = np.divide(new_ndarray, div)
    # np.savetxt(filepath, new_ndarray, fmt='%d')
    new_cloud_array = euler_rotate(new_ndarray, radar)
    all_cloud_list.append(new_cloud_array)
    return new_cloud_array


def euler_rotate(cloud_ndarray, radar, save_path=None):
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

    # 点云旋转操作
    matrix = np.dot(x_matrix, np.dot(y_matrix, z_matrix))
    rotate_cloud_array = np.dot(cloud_ndarray, matrix)

    # 点云平移操作
    shift_xyz = np.array([[radar.shiftX, radar.shiftY, radar.shiftZ]])
    new_cloud_array = rotate_cloud_array + shift_xyz

    # np.savetxt(save_path, new_cloud_array, fmt='%d')
    # print("我要保证在前面 id==", radar.id)
    return new_cloud_array


def put_cloud(filepath, filename):
    object_name = 'cloud_date/' + time.strftime("%Y/%m/%d/") + filename
    # inventory-coal/cloud_date/2022/06/14/a.txt
    minio_conf = settings.MINIO_CONF
    minio_client = minio.Minio(**minio_conf)
    minio_client.fput_object(bucket_name='inventory-coal',
                             object_name=object_name,
                             file_path=filepath,
                             content_type="application/csv")

    minio_path = "http://" + minio_conf['endpoint'] + '/inventory-coal/' + object_name
    print("minio_path == ", minio_path)
    return minio_path


def heap_volume_and_maxheight(coal_heap, cloud_pdarray: DataFrame, minio_path: str, minio_name: str):
    x_list = []
    y_list = []
    for heap_point in coal_heap.coalHeapArea:
        x_list.append(heap_point.x)
        y_list.append(heap_point.y)
        max_x = max(x_list)
        min_x = min(x_list)
        max_y = max(y_list)
        min_y = min(y_list)

    # a = cloud_pdarray
    split_pdarray = cloud_pdarray[(cloud_pdarray['x'] < max_x) & (cloud_pdarray['x'] > min_x)
                                  & (cloud_pdarray['y'] < max_y) & (cloud_pdarray['y'] > min_y)]
    # split_ndarray = split_pdarray.values

    split_pdarray.to_csv(minio_path, index=False, header=False, sep=' ', float_format='%.6f')
    minio_url = put_cloud(minio_path, minio_name)
    print("*****************************")

    split_ndarray = split_pdarray.values
    try:
        u = split_ndarray[:, 0]  # 这里会报错，如果filename为空，即切割区域没有点云
        v = split_ndarray[:, 1]
        z = split_ndarray[:, 2]
        x = u
        y = v
        maxHeight = max(z) - min(z)
    except:
        return {'maxHeight': 0, 'volume': 0, 'minio_url': minio_url}

    ply_name = minio_path.replace('.txt', '.ply')
    tri = Delaunay(np.array([u, v]).T)
    f2 = open(ply_name, 'w')
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
        f2.write('%d %d %d\n' % (x[i], y[i], z[i]))
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
    response = {'maxHeight': maxHeight, 'volume': volume, 'minio_url': minio_url}
    # print('response ==', response)
    return response
