"""
@Time : 2022/6/16 11:48
@Author : ls
@DES:
"""
import os
import io
import time
import numpy
import ctypes
import asyncio
import requests
from core.Response import *
from fastapi import APIRouter
from api.endpoints.coal import inventory_coal as inventory_coal_test
from methods.radar_func import *
from methods.cloud_stent import remove_stents
from methods.cloud_cover import remove_cover_by_list
from methods.cloud_noise import remove_point_cloud_noise
from methods.put_cloud import put_cloud_to_minio
from methods.polygon_filter import is_poi_within_polygon
from methods.bounding_box_filter import bounding_box_filter
from methods.calculate_volume import new_heap_vom_and_maxheight
from models.custom_stent import stents
from models.dict_obj import DictObj
from models.custom_class import CoalYard, CoalRadar, InventoryCoalResult

router = APIRouter()
coal_yard_list: List[CoalYard] = []
polygon = [[8.0, 8.0], [420.0, 8.0], [420.0, 170.0], [8.0, 220.0], [8.0, 8.0]]
split_list = [[10.5, 12.5, 8], [12.5, 14.0, 10], [14.0, 15.0, 11], [15.0, 16.0, 11], [16.0, 17.0, 13],
              [17.0, 18.0, 14], [18.0, 19.0, 16], [19.0, 20.0, 18], [20.0, 21.0, 20], [21.0, 180.0, 26],
              [180.0, 192.0, 10.0], [192.0, 195.0, 8.0], [195.0, 198.0, 6.0], [198.0, 200.0, 5.0],
              [200.0, 202.0, 4.0], [202.0, 204.0, 2.0], [204.0, 208.0, 2.0], [208.0, 212.0, 1.0],
              [212.0, 216.0, 1.0], [216.0, 2220.0, 1.0], [220.0, 240.0, 1.0]]


@router.post("/coal_test_v2", summary="标准测试版")
async def inventory_coal(yard_id: int):
    base_url = settings.PATH_CONFIG.BASEURL
    url = base_url + '/coal/coalYard/realTime/coalYardInfo?coalYardId=' + str(yard_id)
    response = requests.get(url).json()
    # DictOBj将一个dict转化为一个对象，方便以属性的方式访问
    coal_yard_dict = response['data']
    coal_yard: CoalYard = DictObj(coal_yard_dict)
    coal_yard_list.append(coal_yard)

    # 给煤场对象添加属性
    coal_yard.conn_radarsBucket = []
    for radar in coal_yard.coalRadarList:
        radar.bytes_buffer = bytes()

    # 判断煤场id, id=8 or 10,走模拟数据
    # if coal_yard.coalYardId == 8 or coal_yard.coalYardId == 10:
    #     return await inventory_coal_test(coal_yard)
    # 根据煤场id无法在回调中获取coal_yard对象，设置全局coal_yard
    # yard_id = id(coal_yard)

    cloud_ndarray_list: List[numpy.ndarray] = list()
    set_callback_function(func=_callback, obj_id=111)

    radars = coal_yard.coalRadarList
    radars_start_connect(radars=radars)

    await asyncio.sleep(2)
    begin_response = radars_rotate_begin(radars=radars, auto_yard=coal_yard)
    if begin_response is False:
        return fail(msg="存在未连接成功的雷达，启动失败！")

    # 代表全部雷达停止的判断条件， 如果存在未中断连接的雷达则进入循环，否则跳出
    while len(coal_yard.conn_radarsBucket) != 0:
        continue

    radars = coal_yard.coalRadarList
    for radar in radars:
        radar_bytes_data = radar.bytes_buffer
        radar_cloud_ndarray: numpy.ndarray = bytes_cloud_data_rotate_and_shift(bytes_data=radar_bytes_data,
                                                                               radar=radar)
        cloud_ndarray_list.append(radar_cloud_ndarray)

    combined_cloud_ndarray: numpy.ndarray = numpy.concatenate(cloud_ndarray_list, axis=0)

    # 点云去噪操作
    new_cloud: numpy.ndarray = remove_point_cloud_noise(cloud=combined_cloud_ndarray)

    # 保存点云文件操作
    # np.savetxt(fname='combined_cloud.txt', X=new_cloud, fmt='%.5f', delimiter=' ')

    # 去除底面操作
    new_cloud: numpy.ndarray = new_cloud[new_cloud[:, 2] >= 2.0]

    # 去除棚顶和保留多边形边界操作
    # new_cloud: numpy.ndarray = remove_cover_and_bottom(new_cloud, cover=12.0, bottom=-1.0)
    new_cloud: numpy.ndarray = remove_cover_by_list(cloud=new_cloud, s_list=split_list, polygon=polygon)

    # 去除柱子操作
    new_cloud: numpy.ndarray = remove_stents(cloud=new_cloud, stent_list=stents)
    # 多边形切割操作
    # new_cloud: numpy.ndarray = remove_out_polygon_point(new_cloud, poly=polygon)

    # 进行煤堆切割并计算体积
    res_list: List[InventoryCoalResult] = await split_and_calculate_volume(cloud_ndarray=new_cloud)

    return success(data=res_list)


def _callback(cid: c_uint, data_len: c_int, data, yard_id):
    # 根据雷达 id 获取对应的 coal_yard 对象
    coal_yard = get_yard_by_cid(cid=cid)
    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    if code == 3534:
        print("雷达连接成功, cid ==", cid)
        bucket = coal_yard.conn_radarsBucket
        if cid not in bucket:
            bucket.append(cid)
        # RADARS_BUCKET.append(cid)
    elif code == 3535:
        print("连接失败")
        # if cid in bucket:
        #     bucket.remove(cid)
    elif code == 51108:
        print("运行模式设置成功")
    elif code == 118:
        points_data: bytes = data[54:data_len]

        radars = coal_yard.coalRadarList
        for radar in radars:
            if radar.id == cid:
                # 点云数据(bytes类型)循环写入对应radar的属性bytes_buffer中
                radar.bytes_buffer += points_data

        last_line_flag = data[44]
        if last_line_flag == b'\x80':
            print('雷达停止， cid ====================== ', cid)
            # radar_stop 函数停止并关闭雷达连接，同时在RADAR_BUCKET中删除雷达id
            radar_stop(c_id=cid)
            bucket = coal_yard.conn_radarsBucket
            if cid in bucket:
                bucket.remove(cid)
                print('删除雷达， cid ====================== ', cid)
    return


def get_yard_by_cid(cid):
    for coal_yard in coal_yard_list:
        radars = coal_yard.coalRadarList
        for radar in radars:
            if cid == radar.id:
                return coal_yard


def radars_rotate_begin(radars: List[CoalRadar], auto_yard: CoalYard):
    bucket = auto_yard.conn_radarsBucket
    for radar in radars:
        if radar.id not in bucket:
            return False

    # await websocket.send_text('开始盘煤')
    for radar in radars:
        cid = radar.id
        # if cid not in RUNNING_RADARS_BUCKET:
        #     RUNNING_RADARS_BUCKET.append(cid)
        stopResult = dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        if stopResult:
            print(str(cid) + "号雷达停止指令设置成功")
        else:
            print(str(cid) + "号雷达停止指令设置失败")
        runModeResult = dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, RunMode, Speed, 0, AngleSceneScan)
        if runModeResult:
            print(str(cid) + "号雷达模式指令设置成功")
        else:
            print(str(cid) + "号雷达模式指令设置失败")
        beginResult = dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, Speed, 0, AngleSceneScan)
        if beginResult:
            print(str(cid) + "号雷达开始指令设置成功")
        else:
            print(str(cid) + "号雷达开始指令设置失败")
    return True


# 输入自定义对象的id值，返回一个 object 对象
def get_coal_yard_by_id(yard_id: int):
    custom_object = ctypes.cast(yard_id, ctypes.py_object).value
    return custom_object


async def split_and_calculate_volume(cloud_ndarray: numpy.ndarray):
    res_list: List[InventoryCoalResult] = list()  # 设置一个空字典，接收煤堆对象
    time_stamp = str(time.strftime("%m%d%H%M%S"))  # 设置时间戳，标记文件生成时间

    coal_yard = coal_yard_list[0]
    # 判断yard_name 文件夹是否存在，不存在创建
    coal_yard_directory = settings.DATA_PATH + '/' + coal_yard.coalYardName
    if not os.path.exists(coal_yard_directory):
        os.makedirs(coal_yard_directory)

    # 遍历获取单个煤堆信息，并进行计算体积操作
    heaps = coal_yard.coalHeapList
    for heap in heaps:
        res = InventoryCoalResult()
        res.coalHeapId = heap.coalHeapId
        res.coalHeapName = heap.coalHeapName
        res.density = heap.density
        res.mesId = heap.mesId

        minio_name = 'coalHeap' + str(heap.coalHeapId) + '_' + time_stamp + '.txt'
        minio_path = coal_yard_directory + '/' + minio_name

        # 根据煤堆区域切割获取小点云文件(ndarray类型)，并保存ndarray类型为txt文件
        split_cloud_ndarray: numpy.ndarray = bounding_box_filter(cloud_ndarray, heap.coalHeapArea)
        # numpy.savetxt(fname=minio_path, X=split_cloud_ndarray, fmt='%.2f', delimiter=' ')

        list_cloud = split_cloud_ndarray.tolist()
        bytes_cloud = bytes(str(list_cloud), encoding='utf-8')

        # 根据小点云文件(ndarray类型)计算体积和高度
        vom_start = datetime.now()
        vom_and_maxheight = await new_heap_vom_and_maxheight(cloud_ndarray=split_cloud_ndarray, minio_path=minio_path)
        vom_end = datetime.now()
        print(f"{heap.coalHeapName} 计算体积运行时间 === {vom_end - vom_start}")
        res.volume = vom_and_maxheight['volume']
        res.maxHeight = vom_and_maxheight['maxHeight']
        print("%s 体积: %.2f，高度: %.2f" % (res.coalHeapName, res.volume, res.maxHeight))

        # 上传文件至 minio,返回minio文件路径
        # 例如："http://172.16.200.243:9000/inventory-coal/2022/05/24/1_20220510144243A001.txt"
        data_buffer = io.BytesIO(bytes_cloud)
        res.cloudInfo = put_cloud_to_minio(f_name=minio_name, data=data_buffer, length=len(bytes_cloud))

        # 煤堆信息对象保存至 list
        res_list.append(res)
    return res_list


def euler_rotate(cloud_ndarray: numpy.ndarray, radar: CoalRadar):
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
    # shift_xyz = np.array([[radar.shiftX / 100, radar.shiftY / 100, radar.shiftZ / 100]])
    # new_cloud_array = rotate_cloud_array + shift_xyz
    #
    # new_cloud_array.astype(np.float16)
    return rotate_cloud_array


def bytes_cloud_data_rotate_and_shift(bytes_data: bytes, radar: CoalRadar):
    cloud_ndarray = np.frombuffer(bytes_data, dtype=np.int16).reshape(-1, 3)
    div = np.array([100, 100, 100])
    radar_cloud_ndarray = np.divide(cloud_ndarray, div)
    radar_cloud_ndarray.astype(np.float16)

    rotated_radar_cloud_ndarray = euler_rotate(radar_cloud_ndarray, radar=radar)
    rotated_radar_cloud_ndarray = rotated_radar_cloud_ndarray.astype(np.float32)

    # 点云平移操作
    shift_xyz = np.array([radar.shiftX, radar.shiftY, radar.shiftZ])
    new_cloud_array = rotated_radar_cloud_ndarray + shift_xyz
    # 点云乘以-1，适应3d煤场区域
    new_cloud_array = new_cloud_array * numpy.array([[-1, 1, 1]])

    return new_cloud_array

    # save_path = FRAME_DATA_PATH + '/radar_' + str(cid) + '_cloudData_' + str(time_now) + ".txt"
    # np.savetxt(fname=save_path, X=rotated_radar_cloud_ndarray, fmt='%.2f', delimiter=' ')
    # 以上代表已经经过欧拉旋转并且平移的 帧数据
    # return rotated_radar_cloud_ndarray
    # len_array = rotated_radar_cloud_ndarray.__len__()
    # all_cloud_list.append(rotated_radar_cloud_ndarray)


def remove_cover_and_bottom(cloud: numpy.ndarray, cover: float = 12.0, bottom: float = -1.0):
    # 去除弧顶
    max_z = cloud[:, 2] < cover
    min_z = cloud[:, 2] > bottom
    cloud = cloud[min_z & max_z]
    return cloud


def remove_out_polygon_point(cloud: numpy.ndarray, poly: list):
    new_list = []
    rows = cloud.shape[0]
    for row in range(rows):
        poi = cloud[row][0:2].tolist()
        if is_poi_within_polygon(poi, poly):
            a_list = cloud[row]
            new_list.append(a_list)
    new_array = numpy.array(new_list)
    return new_array
