"""
@Time : 2022/5/31 11:47 
@Author : lpy
@DES: 盘煤接口
"""
import io
import os
import time
import numpy
from datetime import datetime
from fastapi import APIRouter
from config import settings
from core.Response import success, fail
from concurrent.futures import ThreadPoolExecutor
from methods.cloud_concatenate import cloud_concatenate
from methods.put_cloud import put_cloud_to_minio
from methods.bounding_box_filter import bounding_box_filter
from models.custom_class import CoalYard, InventoryCoalResult
from methods.get_vom_and_maxheight import heap_vom_and_maxheight

router = APIRouter()


@router.post("/inventoryCoal", summary="模拟数据测试")
async def inventory_coal(coal_yard: CoalYard = None):
    start_time = datetime.now()

    res_list = list()  # 设置一个空字典，接收煤堆对象
    time_stamp = str(time.strftime("%m%d%H%M%S"))

    # 获取雷达参数，并调用雷达获取煤场点云文件
    # radars = coal_yard.coalRadarList
    # txt_cloud_list = radar_start(radars)

    # 判断yard_name 文件夹是否存在，不存在创建
    coal_yard_directory = settings.DATA_PATH + '/' + coal_yard.coalYardName
    if not os.path.exists(coal_yard_directory):
        os.makedirs(coal_yard_directory)

    combined_cloud_directory = settings.CLOUD_COMBINED_PATH
    if not os.path.exists(combined_cloud_directory):
        os.makedirs(combined_cloud_directory)

    # 从模拟数据文件路径取数据，转化为 ndarray 类型
    combined_filename = combined_cloud_directory + "/cloudData_test.txt"
    s_time = datetime.now()
    combined_cloud_ndarray = numpy.genfromtxt(combined_filename)
    e_time = datetime.now()
    run_time = e_time - s_time
    print(f"解析时间 ======{run_time}")

    # 遍历获取单个煤堆信息，并进行计算体积操作
    heaps = coal_yard.coalHeapList
    print('heladlfa')
    for heap in heaps:
        res = InventoryCoalResult()
        res.coalHeapId = heap.coalHeapId
        res.coalHeapName = heap.coalHeapName
        res.density = heap.density
        res.mesId = heap.mesId

        minio_name = 'coalHeap' + str(heap.coalHeapId) + '_' + time_stamp + '.txt'
        minio_path = coal_yard_directory + '/' + minio_name

        # 根据煤堆区域切割获取小点云文件(ndarray类型)，并保存ndarray类型为txt文件
        split_cloud_ndarray: numpy.ndarray = bounding_box_filter(combined_cloud_ndarray, heap.coalHeapArea)
        # numpy.savetxt(fname=minio_path, X=split_cloud_ndarray, fmt='%.2f', delimiter=' ')

        list_cloud = split_cloud_ndarray.tolist()
        bytes_cloud = bytes(str(list_cloud), encoding='utf-8')
        res.bytes_buffer = bytes_cloud

        # 根据小点云文件(ndarray类型)计算体积和高度
        vom_start = datetime.now()
        print("进入体积计算")
        vom_and_maxheight = await heap_vom_and_maxheight(cloud_ndarray=split_cloud_ndarray, minio_path=minio_path)
        vom_end = datetime.now()
        print(f"{heap.coalHeapName} 计算体积运行时间 === {vom_end - vom_start}")
        res.volume = vom_and_maxheight['volume']
        res.maxHeight = vom_and_maxheight['maxHeight']
        print("%s 体积: %.2f，高度: %.2f" % (res.coalHeapName, res.volume, res.maxHeight))

        # 上传文件至 minio,返回minio文件路径
        # 例如："http://172.16.200.243:9000/inventory-coal/2022/05/24/1_20220510144243A001.txt"
        # data_buffer = io.BytesIO(bytes_cloud)
        # res.cloudInfo = await put_cloud_to_minio(f_name=minio_name, data=data_buffer, length=len(bytes_cloud))

        # 煤堆信息对象保存至 list
        res_list.append(res)

    put_put(res_list, coal_yard=coal_yard)

    end_time = datetime.now()
    print("运行时间 ==============", end_time - start_time)
    return success(msg='盘煤成功', data=res_list)


def put_put(res_list, coal_yard):
    print("进入上传 mino ")
    time_stamp = str(time.strftime("%m%d%H%M%S"))

    # 判断yard_name 文件夹是否存在，不存在创建
    coal_yard_directory = settings.DATA_PATH + '/' + coal_yard.coalYardName
    if not os.path.exists(coal_yard_directory):
        os.makedirs(coal_yard_directory)

    for heap in res_list:
        minio_name = 'coalHeap' + str(heap.coalHeapId) + '_' + time_stamp + '.txt'
        minio_path = coal_yard_directory + '/' + minio_name

        data_buffer = io.BytesIO(heap.bytes_buffer)
        put_cloud_to_minio(f_name=minio_name, data=data_buffer, length=len(heap.bytes_buffer))
