"""
@Time : 2022/6/16 11:48
@Author : ls
@DES:
"""
import os
import math
import copy
import threading
import time
import numpy
import minio
import platform
import numpy as np
from ctypes import *
from io import BytesIO
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
from models.custom_class import CoalYard, CoalRadar, InventoryCoalResult
from core.Response import success, fail
from pydantic import BaseModel


router = APIRouter()
queue = Queue()

gCallbackFuncList = []  # 定义全局列表接收回调，防止系统回收
dllPath = settings.STATIC_DIR + '/sdk/CDPSIMCLIENT.dll'
soPath = settings.STATIC_DIR + '/sdk/cdpsimclient-linux64.so'

if platform.system() == 'Windows':
    from ctypes import windll
    dll = windll.LoadLibrary(dllPath)
else:
    from ctypes import cdll
    dll = cdll.LoadLibrary(soPath)


@router.post("/coal_test", summary="刘赛的煤场测试")
async def inventory_coal(CoalYard: CoalYard):
    # 这是最重要的两个版本，代码不能动
    '''
    :param CoalYard: 煤场信息，包含雷达信息，煤堆信息
    :return:
    '''
    return main_function(CoalYard)
    # another_main_function 实现数据实时传输、实时显示
    # return another_main_function(CoalYard)


def _callback(cid: c_uint, datalen: c_int, data, createe_time):
    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    if code == 3534:
        # print("连接成功标志")
        dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, runmode, 64, 0, 360)
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, speed, 0, AngleSceneScan)
        if cid not in conn_radar:
            conn_radar.append(cid)
    elif code == 3535:
        print("连接失败")
        # file.close()
        conn_radar.remove(cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_StopConnectCid(cid)
    elif code == 51108:
        print("运行模式设置成功")
    elif code == 118:
        points_data = data[54:datalen]
        new_points_data = join_cid_to_bytes(point_bytes=points_data, cid=cid)
        bytes_io.write(new_points_data)
        last_line_flag = data[44]
        if last_line_flag == b'\x80':
            conn_radar.remove(cid)
            dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
            dll.NET_SDK_SIMCLT_StopConnectCid(cid)
    else:
        pass
        # print('其他未知码')
    return


def _another_callback(cid: c_uint, datalen: c_int, data, createe_time):
    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    if code == 3534:
        # print("连接成功标志")
        dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, runmode, 64, 0, 360)
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, speed, 0, AngleSceneScan)
        if cid not in conn_radar:
            conn_radar.append(cid)
    elif code == 3535:
        print("连接失败")
        # file.close()
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_StopConnectCid(cid)
        # if cid in conn_radar:
        #     conn_radar.remove(cid)
    elif code == 51108:
        print("运行模式设置成功")
    elif code == 118:
        points_data = data[54:datalen]
        new_points_data = join_cid_to_bytes(point_bytes=points_data, cid=cid)
        # new_points_data 代表一帧数据，总长是8的倍数(一个点占8个字节)
        allData.append(new_points_data)
        # bytes_io.write(new_points_data)

        last_line_flag = data[44]
        if last_line_flag == b'\x80':
            # conn_radar.remove(cid)
            global closedCount
            closedCount += 1
            dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
            dll.NET_SDK_SIMCLT_StopConnectCid(cid)
    else:
        # pass
        print('其他未知码 ==', code)
    return


def join_cid_to_bytes(point_bytes: bytes, cid: int):
    byte_cid = cid.to_bytes(2, 'little')
    len_b = int(len(point_bytes) / 6)

    j = 0
    new_point_bytes = b''
    for i in range(len_b):
        dd = byte_cid + point_bytes[j:j + 6]
        new_point_bytes += dd
        j += 6

    return new_point_bytes


def my_most_valuable_function(cloud: numpy.ndarray, CoalYard: CoalYard):
    tempResList = []
    cloud_pdarray = DataFrame(cloud[:, 0:4])
    cloud_pdarray.columns = ['cid', 'x', 'y', 'z']  # 给选取到的数据 附上标题

    # 判断yard_name 文件夹是否存在，不存在创建
    coal_yard_path = settings.DATA_PATH + '/' + CoalYard.coalYardName
    if not os.path.exists(coal_yard_path):
        os.makedirs(coal_yard_path)

    radar_list = CoalYard.coalRadarList
    for radar in radar_list:
        cid = radar.id
        # radar_cloud_pdarray = cloud_pdarray[cloud_pdarray['cid'] == cid]
        radar_cloud_pdarray = cloud_pdarray[cloud_pdarray['cid'] == cid][['x', 'y', 'z']]
        radar_cloud_ndarray = radar_cloud_pdarray.values

        div = np.array([100, 100, 100])
        radar_cloud_ndarray = np.divide(radar_cloud_ndarray, div)
        radar_cloud_ndarray.astype(np.float16)

        rotated_radar_cloud_ndarray = euler_rotate(radar_cloud_ndarray, radar)
        len_array = rotated_radar_cloud_ndarray.__len__()

        save_path = coal_yard_path + '/radar_' + str(cid) + '_cloudData_' + \
                    str(len_array) + '_time_' + str(create_time) + ".txt"

        np.savetxt(fname=save_path, X=rotated_radar_cloud_ndarray, fmt='%.2f', delimiter=' ')

        all_cloud_list.append(rotated_radar_cloud_ndarray)

    combined_cloud_ndarray = np.concatenate(all_cloud_list, axis=0)
    len_combined = combined_cloud_ndarray.__len__()
    combined_filename = coal_yard_path + '/allcloudData_' + str(len_combined) + '_time_' + \
                        str(create_time) + '.txt'
    np.savetxt(fname=combined_filename, X=combined_cloud_ndarray, fmt='%.2f', delimiter=' ')

    combined_cloud_pdarray = DataFrame(combined_cloud_ndarray[:, 0:3])
    combined_cloud_pdarray.columns = ['x', 'y', 'z']
    combined_cloud_pynt = PyntCloud(combined_cloud_pdarray)
    voxelgrid_id = combined_cloud_pynt.add_structure("voxelgrid", n_x=100, n_y=100, n_z=200)
    # 下面可能会报内存爆满错误 ！！！
    sampled_cloud_pdarray = combined_cloud_pynt.get_sample("voxelgrid_centroids", voxelgrid_id=voxelgrid_id,
                                                           as_PyntCloud=False)
    len_sampled = sampled_cloud_pdarray.__len__()
    sampled_filename = coal_yard_path + "/sampled_cloudData_" + str(len_sampled) \
                       + '_time_' + str(create_time) + ".txt"
    sampled_cloud_pdarray.to_csv(sampled_filename, index=False, header=False, sep=' ', float_format='%.2f')

    for coal_heap in CoalYard.coalHeapList:
        res = InventoryCoalResult()
        res.coalHeapId = coal_heap.coalHeapId
        res.coalHeapName = coal_heap.coalHeapName
        res.density = coal_heap.density
        res.mesId = coal_heap.mesId
        # print('coal_heap ==', coal_heap)
        minio_name = 'coalHeap' + str(coal_heap.coalHeapId) + '_' + str(create_time) + '.txt'
        minio_path = coal_yard_path + '/' + minio_name
        vom_and_maxhei_and_minio = heap_volume_and_maxheight(coal_heap, sampled_cloud_pdarray,
                                                             minio_path=minio_path, minio_name=minio_name)
        # res.cloudInfo = put_cloud(minio_path, minio_name)
        res.cloudInfo = vom_and_maxhei_and_minio['minio_url']
        res.volume = vom_and_maxhei_and_minio['volume']
        res.maxHeight = vom_and_maxhei_and_minio['maxHeight']
        tempResList.append(res)
    return tempResList


def convert_bin_by_radar(bin_filename: str, radar: CoalRadar):
    bin_ndarray = np.fromfile(bin_filename, dtype=np.int16)
    new_ndarray = bin_ndarray.reshape(-1, 3)
    div = np.array([100, 100, 100])
    new_ndarray = np.divide(new_ndarray, div)
    # np.savetxt(filepath, new_ndarray, fmt='%d')
    new_cloud_array = euler_rotate(new_ndarray, radar)
    all_cloud_list.append(new_cloud_array)
    return new_cloud_array


def euler_rotate(cloud_ndarray: numpy.ndarray, radar, save_path=None):
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

    new_cloud_array.astype(np.float16)
    # np.savetxt(save_path, new_cloud_array, fmt='%d')
    # print("我要保证在前面 id==", radar.id)
    return new_cloud_array


def put_cloud(filepath: str, filename: str):
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

    # minio_path 为本地的文件名
    split_pdarray.to_csv(minio_path, index=False, header=False, sep=' ', float_format='%.6f')
    minio_url = put_cloud(minio_path, minio_name)
    print("*****************************")

    split_ndarray = split_pdarray.values
    if split_ndarray.__len__() < 500:
        return {'maxHeight': 0, 'volume': 0, 'minio_url': minio_url}

    u = split_ndarray[:, 0]  # 这里会报错，如果filename为空，即切割区域没有点云
    v = split_ndarray[:, 1]
    z = split_ndarray[:, 2]
    x = u
    y = v
    maxHeight = max(z) - min(z)

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
    print('response ==', response)
    return response


# 实现展示数据
def main_function(CoalYard):
    global resList, all_cloud_list, bytes_io, conn_radar
    resList = list()  # 设置一个空字典，接收煤堆对象
    bytes_io = BytesIO()
    conn_radar = list()
    all_cloud_list = list()
    global runmode, speed, AngleSceneScan, create_time
    runmode = 0
    speed = 20
    AngleSceneScan = 360
    create_time = int(time.strftime('%m%d%H%M%S'))

    init_status = dll.NET_SDK_SIMCLT_Init()
    print("初始化结果:", init_status)
    CALLBACK = WINFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)  # 设置值回调函数
    # CALLBACK = CFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    callBackFunc = CALLBACK(_callback)
    gCallbackFuncList.append(callBackFunc)
    dll.NET_SDK_SIMCLT_Set_Callback(callBackFunc, create_time)

    radars = CoalYard.coalRadarList
    for radar in radars:
        cid = radar.id
        ip = bytes(radar.ip, encoding='utf-8')
        port = radar.port
        res = dll.NET_SDK_SIMCLT_StartConnect(cid, ip, port, 120)
        print('连接状态:', res, 'cid ==', cid)

    time.sleep(settings.CONNECT_ERROR_TIME)  # ? 加入某种方法，判断全部雷达是否全都停止
    if len(conn_radar) == 0:
        return fail(msg='雷达启动失败，网络故障或电源故障')
    while len(conn_radar) != 0:
        # print('雷达正在运行，连接的雷达有:', conn_radar)
        continue

    dll.NET_SDK_SIMCLT_Destory()
    print("===========销毁sdk============")
    startTime = datetime.now()
    bytes_cloud_buffer = bytes_io.getvalue()
    cloud_ndarray = np.frombuffer(bytes_cloud_buffer, dtype=np.int16).reshape(-1, 4)
    my_most_valuable_function(cloud=cloud_ndarray, CoalYard=CoalYard)
    endTime = datetime.now()
    print('点云旋转平移计算体积去重稀释完成，耗时==', endTime - startTime)

    return success(msg='盘煤成功', data=resList)


def get_value(CoalYard):
    while True:
        if len(conn_radar) == closedCount and closedCount > 0:
            break
        if allData.__len__() > 0:
            copyData = copy.deepcopy(allData)
            for a in copyData:
                allData.remove(a)

            print("处理 copyData 里的数据")
            bbb = b''
            for cd in copyData:
                bbb += cd

            cloud_ndarray = np.frombuffer(bbb, dtype=np.int16).reshape(-1, 4)
            resList = []
            resList = my_most_valuable_function(cloud=cloud_ndarray, CoalYard=CoalYard)
            # 返回给用户
            print(resList)
        time.sleep(0.2)


def another_main_function(CoalYard):
    global resList, all_cloud_list, bytes_io, conn_radar, allData
    bytes_io = BytesIO()
    allData = list()
    resList = list()  # 设置一个空字典，接收煤堆对象
    conn_radar = list()
    all_cloud_list = list()
    global runmode, speed, AngleSceneScan, create_time, closedCount
    runmode = 0
    speed = 20
    AngleSceneScan = 360
    closedCount = 0
    create_time = int(time.strftime('%m%d%H%M%S'))

    init_status = dll.NET_SDK_SIMCLT_Init()
    print("初始化结果:", init_status)
    CALLBACK = WINFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)  # 设置值回调函数
    # CALLBACK = CFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    callBackFunc = CALLBACK(_another_callback)
    gCallbackFuncList.append(callBackFunc)
    dll.NET_SDK_SIMCLT_Set_Callback(callBackFunc, create_time)

    radars = CoalYard.coalRadarList
    for radar in radars:
        cid = radar.id
        ip = bytes(radar.ip, encoding='utf-8')
        port = radar.port
        res = dll.NET_SDK_SIMCLT_StartConnect(cid, ip, port, 120)
        print('连接状态:', res, 'cid ==', cid)

    # time.sleep(0.1)
    # if len(conn_radar) == 0:
    #     return fail(msg='雷达启动失败，网络故障或电源故障')

    read_thread = threading.Thread(target=get_value, args=(CoalYard,))
    read_thread.start()
    read_thread.join()

    return success(msg='盘煤成功', data=resList)
