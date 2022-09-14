"""
@Time : 2022/9/14 11:06 
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
from methods.radar_func import *
from methods.cloud_cover import remove_cover
from methods.cloud_stent import remove_stents
from methods.cloud_noise import remove_noise
from methods.put_cloud import put_ply_to_minio
from methods.put_cloud import put_cloud_to_minio
from methods.cloud_save import save_cloud
from methods.cloud_transform import radars_cloud_transform
from methods.polygon_filter import is_poi_within_polygon
from methods.bounding_box_filter import bounding_box_filter
from methods.cloud_volume import *
from models.custom_stent import stents
from models.dict_obj import DictObj
from models.custom_class import CoalYard, CoalRadar, InventoryCoalResult

from multiprocessing import Process, Manager, active_children
import random
import time

coal_yard_list = []


class CloudProducer(Process):

    def __init__(self, queue, coal_yard):
        super().__init__()
        self.queue = queue
        self.coal_yard = coal_yard
        self.sdk_init()
        self.set_callback_function(func=_callback, addr_id=111)

    def run(self):
        for i in range(6):
            r = random.randint(0, 99)
            time.sleep(1)
            self.queue.put(r)
            print("将 {} 放入管道".format(r))

    @staticmethod
    def sdk_init():
        print("执行初始化")
        dll.NET_SDK_SIMCLT_Init()

    @staticmethod
    def set_callback_function(func, addr_id):
        print("执行设置回调")
        call_back = CALLBACK(func)
        gCallbackFuncList.append(call_back)
        dll.NET_SDK_SIMCLT_Set_Callback(call_back, addr_id)


def invent_coal(self, coal_yard: CoalYard):
    coal_yard_list.append(coal_yard)

    cloud_ndarray_list: List[numpy.ndarray] = list()
    set_callback_function(func=_callback, obj_id=111)

    radars = coal_yard.coalRadarList
    radars_start_connect(radars=radars)

    time.sleep(2)
    # await asyncio.sleep(2)

    begin_response = radars_rotate_begin(radars=radars, auto_yard=coal_yard)
    if begin_response is False:
        return fail(msg="存在未连接成功的雷达，启动失败！")

    # 代表全部雷达停止的判断条件， 如果存在未中断连接的雷达则进入循环，否则跳出
    while len(coal_yard.conn_radarsBucket) != 0:
        continue

    print("雷达运行结束")


class CloudConsumer(Process):

    def __init__(self, queue: Manager().Queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                print("把 {} 取出管道".format(data))


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


if __name__ == '__main__':
    Queue = Manager().Queue()  # 创建队列
    p = CloudProducer(Queue)
    c = CloudConsumer(Queue)
    p.start()
    c.start()
    print(active_children())  # 查看现有的进程
    p.join()
    c.join()
    print("结束")
