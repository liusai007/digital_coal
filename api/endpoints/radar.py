"""
@Time : 2022/5/26 14:21 
@Author : lpy
@DES: 雷达管理
"""

import os
import time
import platform
from config import settings
from fastapi import APIRouter
from typing import List
from models.custom_class import CoalRadar
from core.Response import success, fail
from pydantic import BaseModel
from methods.bytes_to_txt import write_bytes_to_txt
from ctypes import *

router = APIRouter()
dllPath = os.path.join(os.getcwd(), "static/sdk/CDPSIMCLIENT.dll")
sopath = os.path.join(os.getcwd(), "static/sdk/cdpsimclient-linux64.so")

if platform.system() == 'Windows':
    from ctypes import windll  # type: ignore
    dll = windll.LoadLibrary(dllPath)
else:
    from ctypes import cdll
    dll = cdll.LoadLibrary(sopath)

# dll = windll.LoadLibrary(dllPath)
gCallbackFuncList = []


def _callback(cid: c_uint, datalen: c_int, data, create_time: c_int):
    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    if code == 3534:
        print("转台雷达连接成功,当前 cid == ", cid)
        dll.NET_SDK_SIMCLT_ZTRD_SetRunMode(cid, runmode, 20, 0, 360)
        print("转台雷达设置模式成功,mode == %d, cid == %d" % (runmode, cid))
        dll.NET_SDK_SIMCLT_ZTRD_RotateStop(cid)
        print("转台雷达停止成功,当前 cid == ", cid)
        dll.NET_SDK_SIMCLT_ZTRD_RotateBegin(cid, speed, 0, AngleSceneScan)
        print("转台雷达启动成功,speed == %d, cid == %d" % (speed, cid))
    elif code == 3535:
        print("连接失败")
    elif code == 51108:
        print("运行模式设置成功")
    elif code == 118:
        filepath = settings.DATA_PATH + '/' + str(cid)

        if not os.path.exists(path=filepath):
            os.makedirs(filepath)
        filename = filepath + '/cloudData_' + str(create_time) + '.cld'
        # print('============================')
        file = open(filename, mode='ab+')
        # print("文件流对象 == ", f)
        # print('============================')

        bytes_3d = data[54:datalen]
        file.write(bytes_3d)  # 保存bytes码到文件流对象f, f为全局变量
        file.close()

        lastLineFlag = data[44]
        if lastLineFlag == b'\x80':
            print("====%d 号雷达最后一条线，断开连接" % (cid))
            dll.NET_SDK_SIMCLT_StopConnectCid(cid)
            # 设置自定义的回调函数
            txt_cloud_path = write_bytes_to_txt(filename)
            txt_cloud_list.append([cid, txt_cloud_path])
    else:
        print("其他code:" + str(code))
    return


def radar_start(radars: List[CoalRadar]):
    create_time = int(time.strftime('%m%d%H%M%S'))
    init_status = dll.NET_SDK_SIMCLT_Init()
    print("初始化结果:", init_status)

    # 设置值回调函数
    # CALLBACK = WINFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    CALLBACK = CFUNCTYPE(None, c_uint, c_int, POINTER(c_char), c_int)
    callBackFunc = CALLBACK(_callback)
    gCallbackFuncList.append(callBackFunc)
    dll.NET_SDK_SIMCLT_Set_Callback(callBackFunc, create_time)

    global txt_cloud_list, runmode, speed, AngleSceneScan
    txt_cloud_list = []
    runmode = 0
    speed = 64
    AngleSceneScan = 360
    for radar in radars:
        cid = radar.id
        ip = bytes(radar.ip, encoding='utf-8')
        port = radar.port
        dll.NET_SDK_SIMCLT_StartConnect(cid, ip, port, 120)

    for i in range(10000):
        if len(txt_cloud_list) == len(radars):
            break
        time.sleep(0.5)

    return txt_cloud_list
