"""
@Time : 2022/5/31 11:47 
@Author : lpy
@DES: 盘煤接口
"""
import os
import time
from fastapi import APIRouter
from config import settings
from core.Response import success, fail
from methods.cloud_concatenate import cloud_concatenate
from methods.put_cloud import put_cloud
from methods.bounding_box_filter import bounding_box_filter
from models.custom_class import CoalYard, InventoryCoalResult
from methods.get_vom_and_maxheight import get_vom_and_maxheight
from api.endpoints.radar import radar_start
import random

router = APIRouter()


@router.post("/inventoryCoal", summary="煤场盘煤")
async def inventory_coal(coal_yard: CoalYard = None):
    """
    煤场盘煤
    :return:
    """
    resList = list()  # 设置一个空字典，接收煤堆对象
    time_stamp = str(time.strftime("%m%d%H%M%S"))
    # 雷达信息
    radars = coal_yard.coalRadarList
    txt_cloud_list = radar_start(radars)
    # 调用雷达获取煤场点云文件

    combined_cloud_dir = settings.CLOUD_COMBINED_PATH

    if not os.path.exists(combined_cloud_dir):
        os.makedirs(combined_cloud_dir)

    combined_filename = combined_cloud_dir + "/combined_cloud_" + time_stamp + ".txt"

    combined_cloud_ndarray = cloud_concatenate(txt_cloud_list, radars, save_path=combined_filename)

    # 判断yard_name 文件夹是否存在，不存在创建
    if not os.path.exists(coal_yard.coalYardName):
        os.makedirs(coal_yard.coalYardName)
    # 煤堆信息
    heaps = coal_yard.coalHeapList
    for heap in heaps:
        res = InventoryCoalResult()
        res.coalHeapId = heap.coalHeapId
        res.coalHeapName = heap.coalHeapName
        res.density = heap.density
        res.mesId = heap.mesId
        # 根据煤堆区域切割获取小点云文件并计算体积和最大高度
        minio_name = 'coalHeap' + str(heap.coalHeapId) + '_' + time_stamp + '.txt'
        minio_path = coal_yard.coalYardName + '/' + minio_name
        bounding_box_filter(combined_cloud_ndarray, heap.coalHeapArea, save_url=minio_path)
        print("开始进行点云切割操作")
        # cloudURL,heap.coalHeapArea 作为参数，计算体积和高度
        vom_and_maxhei = get_vom_and_maxheight(minio_path)
        # res.volume = round(random.uniform(100, 1000), 2号雷达)
        res.volume = vom_and_maxhei['volume']
        # res.cloudInfo = "http://172.16.200.243:9000/inventory-coal/2022/05/24/1_20220510144243A001.txt"
        # res.cloudInfo = minio_path
        # res.maxHeight = random.randint(20, 50)
        res.maxHeight = vom_and_maxhei['maxHeight']
        print("%s 体积: %.2f" % (res.coalHeapName, res.volume))
        res.cloudInfo = put_cloud(minio_path, minio_name)
        resList.append(res)
    # return resList
    return success(msg='盘煤成功', data=resList)
