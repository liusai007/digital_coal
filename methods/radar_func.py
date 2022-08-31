import math
import time

import numpy
import platform
import numpy as np
from ctypes import *
from config import settings
from datetime import datetime
import scipy.linalg as linalg
from models.custom_class import *
from concurrent.futures import ThreadPoolExecutor
from decimal import *

# 雷达参数设置
RunMode = settings.RADAR.RUNMODE
Speed = settings.RADAR.SPEED
AngleSceneScan = settings.RADAR.ANGLESCENESCAN

ALL_DATA = list()
WEBSOCKET_CLIENTS = list()
gCallbackFuncList = list()  # 定义全局列表接收回调，防止系统回收
callback_time = datetime.now()
RUNNING_RADARS_BUCKET = list()  # 用于存储已经启动并且未停止的雷达
pool = ThreadPoolExecutor(5)  # 不指定数字默认为 cpu_count（CPU数量） + 4
# 上面的代码执行之后就会立刻创建五个等待工作的线程

dllPath = settings.STATIC_DIR + '/sdk/CDPSIMCLIENT.dll'
soPath = settings.STATIC_DIR + '/sdk/cdpsimclient-linux64.so'
if platform.system() == 'Windows':
    from ctypes import windll

    dll = windll.LoadLibrary(dllPath)
    CALLBACK = WINFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    dll.NET_SDK_SIMCLT_Init()
else:
    from ctypes import cdll

    dll = cdll.LoadLibrary(soPath)
    CALLBACK = CFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    dll.NET_SDK_SIMCLT_Init()


# def radar_callback(cid: c_uint, datalen: c_int, data, ws_id):
#     websocket = get_websocket_by_wsid(ws_id=ws_id)
#     bucket = websocket.conn_radarsBucket
#     list_buffer = websocket.listBuffer
#     # print(f'bucket ===== {bucket}')
#     code = int.from_bytes(data[2:4], byteorder='little', signed=True)
#
#     if code == 3534:
#         print("雷达连接成功, cid ==", cid)
#         if cid not in bucket:
#             bucket.append(cid)
#         # RADARS_BUCKET.append(cid)
#     elif code == 3535:
#         print("连接失败")
#         if cid in bucket:
#             bucket.remove(cid)
#     elif code == 51108:
#         print("运行模式设置成功")
#     elif code == 118:
#         global callback_time
#         callback_time = datetime.now()
#
#         points_data = data[54:datalen]
#         # bytes_frame = join_cid_to_bytes(point_bytes=points_data, cid=cid)
#         # bytes_frame代表一帧数据，总长是8的倍数(一个点占8个字节)
#
#         # 设置线程池，将帧数据进行转换计算
#         kwargs = {'data': points_data, 'cid': cid, 'ws_id': ws_id}
#         # pool.submit(bytes_cloud_frame_rotated, kwargs)
#         # cloud_rotated_result = pool.submit(bytes_cloud_frame_rotated, bytes_frame).result()
#         new_cloud_list = bytes_cloud_frame_rotated(kwargs)
#         list_buffer.extend(new_cloud_list)
#
#         last_line_flag = data[44]
#         if last_line_flag == b'\x80':
#             # radar_stop 函数停止并关闭雷达连接，同时在RADAR_BUCKET中删除雷达id
#             radar_stop(c_id=cid)
#             # websocket对象的属性bucket中删除雷达id
#             if cid in bucket:
#                 bucket.remove(cid)
#     else:
#         print('其他未知码 == ', code)
#     return


def set_callback_function(func, obj_id):
    callBackFunc = CALLBACK(func)
    gCallbackFuncList.append(callBackFunc)
    dll.NET_SDK_SIMCLT_Set_Callback(callBackFunc, obj_id)


def bytes_cloud_frame_rotated(kwargs: dict):
    ws_id = kwargs['ws_id']
    websocket = get_websocket_by_wsid(ws_id=ws_id)
    coal_yard = websocket.coalYard
    # list_buffer = websocket.listBuffer

    # bytes_frame = join_cid_to_bytes(point_bytes=points_data, cid=cid)
    # # # 判断yard_name 文件夹是否存在，不存在创建
    # FRAME_DATA_PATH = settings.DATA_PATH + '/' + coal_yard.coalYardName + '/frame_data'
    # if not os.path.exists(FRAME_DATA_PATH):
    #     os.makedirs(FRAME_DATA_PATH)

    points_data = kwargs['data']
    cloud_ndarray: numpy.ndarray = np.frombuffer(points_data, dtype=np.int16).reshape(-1, 3)

    cid = kwargs['cid']
    radars = coal_yard.coalRadarList
    for radar in radars:
        if radar.id == cid:
            # if cloud_pdarray['cid'][0] == cid:
            #     radar_cloud_pdarray = cloud_pdarray[cloud_pdarray['cid'] == cid][['x', 'y', 'z']]
            #     radar_cloud_ndarray = radar_cloud_pdarray.values

            div = np.array([100, 100, 100])
            # div = np.array([1, 1, 1])
            radar_cloud_ndarray = np.divide(cloud_ndarray, div)
            # radar_cloud_ndarray = radar_cloud_ndarray.astype(np.float16)

            rotated_radar_cloud_ndarray: numpy.ndarray = euler_rotate(radar_cloud_ndarray, radar)
            # new_cloud: numpy.ndarray = rotated_radar_cloud_ndarray.reshape(-1, 3)
            # new_cloud: numpy.ndarray = new_cloud[:, 1:4]
            new_cloud_list = rotated_radar_cloud_ndarray.tolist()
            # list_buffer.append(new_cloud_list)
            # list_buffer.extend(new_cloud_list)
            return new_cloud_list
            # await websocket.send_text(str(new_cloud_list))

            # save_path = FRAME_DATA_PATH + '/radar_' + str(cid) + '_cloudData_' + str(time_now) + ".txt"
            # np.savetxt(fname=save_path, X=rotated_radar_cloud_ndarray, fmt='%.2f', delimiter=' ')
            # f = open(save_path, 'r')
            # con = f.readlines()
            # f.close()
            # 以上代表已经经过欧拉旋转并且平移的 帧数据
            # return rotated_radar_cloud_ndarray
            # len_array = rotated_radar_cloud_ndarray.__len__()
            # ALL_DATA.append(con)


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


def euler_rotate(cloud_ndarray: numpy.ndarray, radar: CoalRadar, save_path=None):
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

    s = Decimal('100')
    # 点云平移操作
    shift_xyz = np.array([[float(Decimal(radar.shiftX) / s), float(Decimal(radar.shiftY) / s), float(Decimal(radar.shiftZ) / s)]])
    new_cloud_array = rotate_cloud_array + shift_xyz

    # new_cloud_array.astype(np.float16)
    # np.savetxt(save_path, new_cloud_array, fmt='%d')
    # print("我要保证在前面 id==", radar.id)
    return new_cloud_array


def radars_start_connect(radars):
    for radar in radars:
        cid = radar.id
        ip = bytes(radar.ip, encoding='utf-8')
        port = radar.port
        dll.NET_SDK_SIMCLT_StartConnect(cid, ip, port, 120)
        # print('雷达已连接， cid ==', cid)


def is_every_radar_stop(radars):
    # 判断参数中的雷达是否全部停止，如果存在未停止的雷达，则返回 false
    flag = True
    for radar in radars:
        if radar.id in RUNNING_RADARS_BUCKET:
            flag = False
    return flag


# 输入websocket的id值，返回一个 websocket 对象
def get_websocket_by_wsid(ws_id: int):
    websocket = cast(ws_id, py_object).value
    return websocket


def radar_stop(c_id):
    dll.NET_SDK_SIMCLT_ZTRD_RotateStop(c_id)
    if c_id in RUNNING_RADARS_BUCKET:
        RUNNING_RADARS_BUCKET.remove(c_id)
    dll.NET_SDK_SIMCLT_StopConnectCid(c_id)


def radars_rotate_begin(radars, websocket):
    bucket = websocket.conn_radarsBucket
    for radar in radars:
        if radar.id not in bucket:
            return False

    # await websocket.send_text('开始盘煤')
    for radar in radars:
        cid = radar.id
        if cid not in RUNNING_RADARS_BUCKET:
            RUNNING_RADARS_BUCKET.append(cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, RunMode, 64, 0, 360)
        dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, Speed, 0, AngleSceneScan)

    return True
