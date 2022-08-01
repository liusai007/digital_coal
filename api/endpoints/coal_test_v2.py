"""
@Time : 2022/6/16 11:48
@Author : ls
@DES:
"""
import io
import time
import minio
import asyncio
from core.Response import *
from fastapi import APIRouter
from methods.radar_func import *
from methods.put_cloud import put_cloud_to_minio
from methods.bounding_box_filter import bounding_box_filter
from methods.get_vom_and_maxheight import heap_vom_and_maxheight
from models.custom_class import CoalYard, CoalRadar, InventoryCoalResult

router = APIRouter()


@router.post("/coal_test_v2", summary="标准测试版")
async def inventory_coal(coal_yard: CoalYard):
    yard_id = id(coal_yard)
    cloud_ndarray_list = list()
    set_callback_function(func=_callback, obj_id=yard_id)

    radars = coal_yard.coalRadarList
    radars_start_connect(radars=radars)

    await asyncio.sleep(2)
    begin_response = radars_rotate_begin(radars=radars, coal_yard=coal_yard)
    if begin_response is False:
        return fail(msg="存在未连接成功的雷达，启动失败！")

    # 代表全部雷达停止的判断条件
    while len(coal_yard.conn_radarsBucket) == 0:
        radars = coal_yard.coalRadarList
        for radar in radars:
            radar_bytes_data = radar.bytes_buffer
            radar_cloud_ndarray = bytes_cloud_data_rotated(bytes_data=radar_bytes_data, radar=radar)
            cloud_ndarray_list.append(radar_cloud_ndarray)

        combined_cloud_ndarray = numpy.concatenate(cloud_ndarray_list, axis=0)
        res_list = await split_and_calculate_volume(coal_yard=coal_yard, cloud_ndarray=combined_cloud_ndarray)
        return res_list


def _callback(cid: c_uint, data_len: c_int, data, yard_id):
    coal_yard: CoalYard = get_coal_yard_by_id(yard_id=yard_id)
    bucket = coal_yard.conn_radarsBucket

    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    if code == 3534:
        print("雷达连接成功, cid ==", cid)
        # 根据yard_id获取coal_yard对象
        if cid not in bucket:
            bucket.append(cid)
        # RADARS_BUCKET.append(cid)
    elif code == 3535:
        print("连接失败")
        if cid in bucket:
            bucket.remove(cid)
    elif code == 51108:
        print("运行模式设置成功")
    elif code == 118:
        points_data = data[54:data_len]

        radars = coal_yard.coalRadarList
        for radar in radars:
            if radar.id == cid:
                radar.bytes_buffer += points_data

        last_line_flag = data[44]
        if last_line_flag == b'\x80':
            # radar_stop 函数停止并关闭雷达连接，同时在RADAR_BUCKET中删除雷达id
            radar_stop(c_id=cid)
            if cid in bucket:
                bucket.remove(cid)
    else:
        print('其他未知码 == ', code)
    return


def radars_rotate_begin(radars, coal_yard):
    bucket = coal_yard.conn_radarsBucket
    for radar in radars:
        if radar.id not in bucket:
            return False

    # await websocket.send_text('开始盘煤')
    for radar in radars:
        cid = radar.id
        if cid not in RADARS_BUCKET:
            RADARS_BUCKET.append(cid)
        dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, RunMode, 64, 0, 360)
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, Speed, 0, AngleSceneScan)

    return True


# 输入coal_yard的id值，返回一个 coal_yard 对象
def get_coal_yard_by_id(yard_id: int):
    coal_yard: CoalYard = cast(yard_id, py_object).value
    return coal_yard


async def split_and_calculate_volume(coal_yard, cloud_ndarray: numpy.ndarray):
    res_list = list()  # 设置一个空字典，接收煤堆对象
    time_stamp = str(time.strftime("%m%d%H%M%S"))  # 设置时间戳，标记文件生成时间

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
        vom_and_maxheight = await heap_vom_and_maxheight(cloud_ndarray=split_cloud_ndarray, minio_path=minio_path)
        vom_end = datetime.now()
        print(f"{heap.coalHeapName} 计算体积运行时间 === {vom_end - vom_start}")
        res.volume = vom_and_maxheight['volume']
        res.maxHeight = vom_and_maxheight['maxHeight']
        print("%s 体积: %.2f，高度: %.2f" % (res.coalHeapName, res.volume, res.maxHeight))

        # 上传文件至 minio,返回minio文件路径
        # 例如："http://172.16.200.243:9000/inventory-coal/2022/05/24/1_20220510144243A001.txt"
        data_buffer = io.BytesIO(bytes_cloud)
        res.cloudInfo = await put_cloud_to_minio(f_name=minio_name, data=data_buffer, length=len(bytes_cloud))

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
    shift_xyz = np.array([[radar.shiftX, radar.shiftY, radar.shiftZ]])
    new_cloud_array = rotate_cloud_array + shift_xyz

    new_cloud_array.astype(np.float16)
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


def bytes_cloud_data_rotated(bytes_data: bytes, radar: CoalRadar):
    cloud_ndarray = np.frombuffer(bytes_data, dtype=np.int16).reshape(-1, 4)
    div = np.array([100, 100, 100])
    radar_cloud_ndarray = np.divide(cloud_ndarray, div)
    radar_cloud_ndarray.astype(np.float16)

    rotated_radar_cloud_ndarray = euler_rotate(radar_cloud_ndarray, radar=radar)
    return rotated_radar_cloud_ndarray

    # save_path = FRAME_DATA_PATH + '/radar_' + str(cid) + '_cloudData_' + str(time_now) + ".txt"
    # np.savetxt(fname=save_path, X=rotated_radar_cloud_ndarray, fmt='%.2f', delimiter=' ')
    # 以上代表已经经过欧拉旋转并且平移的 帧数据
    # return rotated_radar_cloud_ndarray
    # len_array = rotated_radar_cloud_ndarray.__len__()
    # all_cloud_list.append(rotated_radar_cloud_ndarray)
