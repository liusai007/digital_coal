"""
@Time : 2022/5/26 10:44
@Author : lpy
@DES:
"""
import _ctypes
import random
import json
import io
import ctypes
import asyncio
import requests
from methods.radar_func import *
from methods.volume_func import *
from methods.send_data import send_frame_data
# from methods.radar_func import radar_callback
from methods.put_cloud import put_cloud_to_minio
from methods.bounding_box_filter import bounding_box_filter
from methods.calculate_volume import heap_vom_and_maxheight
from models.dict_obj import DictObj
from fastapi import WebSocket, WebSocketDisconnect
from api.endpoints.ws import *
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.openapi.docs import (get_redoc_html, get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html)
from config import settings
from core import Events, Exceptions, Middleware, Router
from methods.coal_yard import coal_yard_dict  # 测试使用

application = FastAPI(
    debug=settings.APP_DEBUG,
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url=None,
    redoc_url=None
)

# 事件监听
application.add_event_handler("startup", Events.startup(application))
application.add_event_handler("shutdown", Events.stopping(application))

# 异常错误处理
application.add_exception_handler(HTTPException, Exceptions.http_error_handler)
application.add_exception_handler(RequestValidationError, Exceptions.http422_error_handler)
application.add_exception_handler(Exceptions.UnicornException, Exceptions.unicorn_exception_handler)

# 中间件
application.add_middleware(Middleware.Middleware)

application.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

application.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    session_cookie=settings.SESSION_COOKIE,
    max_age=settings.SESSION_MAX_AGE
)
# 路由
application.include_router(Router.router)

# 静态资源目录
application.mount('/static', StaticFiles(directory=settings.STATIC_DIR), name="static")


# application.state.views = Jinja2Templates(directory=settings.TEMPLATE_DIR)


@application.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=application.openapi_url,
        title=application.title + " - Swagger UI",
        oauth2_redirect_url=application.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
    )


@application.get(application.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@application.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=application.openapi_url,
        title=application.title + " - ReDoc",
        redoc_js_url="/static/swagger/redoc.standalone.js",
    )


@application.get("/")
async def get():
    return HTMLResponse(html)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@application.websocket("/inventory/realTime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    clientId = random.randint(1, 10000)
    websocket.clientId = clientId
    await websocket.send_text("当前websocket的clientId为：" + str(websocket.clientId))
    ws_id = id(websocket)
    await websocket.send_text('Websocket已连接成功，正在等待用户指令！')
    websocket.list_buffer = list()
    base_url = settings.PATH_CONFIG.BASEURL
    # 设置回调函数，将 websocket 内存地址作为参数，传入回调中
    set_callback_function(func=radar_callback, obj_id=websocket.clientId)
    try:
        while True:
            data = await websocket.receive_text()
            yard_id = websocket.query_params['coalYardId']
            # yard_id = 11
            if data == 'start':
                await websocket.send_text("接受的CoalYardId为：" + str(yard_id))
                await websocket.send_text("正在通过指令进行传输，请稍后...")
                url = base_url + '/coal/coalYard/realTime/coalYardInfo?coalYardId=' + str(yard_id)
                response = requests.get(url).json()
                # DictOBj将一个dict转化为一个对象，方便以属性的方式访问
                coal_yard_dict = response['data']
                coal_yard_obj = DictObj(coal_yard_dict)
                logger.info(response)
                # 这一步很重要: 给websocket添加煤场属性，值为煤场对象
                websocket.coalYard = coal_yard_obj
                # websocket.__setattr__('coalYard', coal_yard_obj)
                # 判断是否正在盘煤（如果请求参数中的雷达没有全部停止，则发送等待信号）
                radars = coal_yard_obj.coalRadarList
                if is_every_radar_stop(radars) is False:
                    # 如果雷达没有全部停止，则发送等待信号
                    await websocket.send_text('雷达正在运行中，请稍后重试......')
                elif is_every_radar_stop(radars) is True:
                    websocket.conn_radarsBucket = list()
                    # websocket.__setattr__('conn_radarsBucket', list())
                    # 如果雷达全部处于停止状态，则开始进行连接雷达操作
                    radars_start_connect(radars=radars)
                    await websocket.send_text('开始盘煤')
                    # 设置空list用于在回调函数中保存雷达帧数据
                    # websocket.list_buffer = list()
                    # websocket.__setattr__('list_buffer', list())
                    logger.info("开始睡眠")
                    await asyncio.sleep(2)
                    logger.info("connectBuckets内数据为：")
                    logger.info(websocket.conn_radarsBucket)
                    begin_response = radars_rotate_begin(radars, websocket)
                    logger.info("开始状态")
                    logger.info(begin_response)
                    if begin_response is False:
                        await websocket.send_text('存在未连接成功的雷达，操作中止......')
                        logger.info("存在未连接成功的雷达，操作中止")
                        continue
                # await start(ws_id=ws_id)
            else:
                await websocket.send_text(data='输入的指令暂时无法识别，请联系管理员')
                logger.info("输入的指令暂时无法识别，请联系管理员")
                continue

            # 发送长度为3000的数据帧，n代表一次发送的数据长度
            await websocket.send_text("开始传输数据")
            result = await send_frame_data(websocket, n=3000)
            if result == "success":
                cloud_list = websocket.list_buffer
                cloud_ndarray = numpy.array(cloud_list)
                # 实现用于切割煤场并计算体积的功能
                res_list = await split_and_calculate_volume(coal_yard=coal_yard_obj, cloud_ndarray=cloud_ndarray)
            for i in res_list:
                await websocket.send_text(
                    "煤堆：" + i.coalHeapName + "\n" + "体积：" + str(i.volume) + "\n" + "高度：" + str(i.maxHeight))

            # 实时推送完成后 数据推送到后台接口 POST形式

            coalYardObj = dict()
            coalYardObj["coalYardId"] = coal_yard_obj.coalYardId
            coalYardObj["coalYardName"] = coal_yard_obj.coalYardName
            coalYardObj["inventoryTime"] = datetime.now().strftime('%Y-%m-%d %H:%I:%S')
            coalYardObj["coalHeapResultList"] = res_list

            coalYardObj = json.dumps(coalYardObj, default=convert2json)
            coalYardObj = json.loads(coalYardObj)

            url = base_url + '/coal/coalYard/realTime/inventoryCoalCallback'
            re = requests.post(url, json=coalYardObj).json()
            print("回调结果：" + str(re))
            await websocket.send_text("盘煤回调接口调用成功！")


    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # await manager.broadcast(f"Client #{client_id} left the chat")


def convert2json(person):
    return {
        'coalHeapId': person.coalHeapId,
        'coalHeapName': person.coalHeapName,
        'volume': person.volume,
        'maxHeight': person.maxHeight,
        'density': person.density,
        'mesId': person.mesId
    }


def radar_callback(cid: c_uint, datalen: c_int, data, ws_id):
    for i in manager.active_connections:
        if ws_id == i.clientId:
            logger.info("进入了正确的websocket判断区间")
            websocket = i
        else:
            websocket = manager.active_connections[0]

    bucket = websocket.conn_radarsBucket
    list_buffer = websocket.list_buffer
    # print(f'bucket ===== {bucket}')
    code = int.from_bytes(data[2:4], byteorder='little', signed=True)

    if code == 3534:
        print("雷达连接成功, cid ==", cid)
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
        global callback_time
        callback_time = datetime.now()

        points_data = data[54:datalen]
        # bytes_frame = join_cid_to_bytes(point_bytes=points_data, cid=cid)
        # bytes_frame代表一帧数据，总长是8的倍数(一个点占8个字节)

        # 设置线程池，将帧数据进行转换计算
        kwargs = {'data': points_data, 'cid': cid, 'ws_id': ws_id}
        # pool.submit(bytes_cloud_frame_rotated, kwargs)
        # cloud_rotated_result = pool.submit(bytes_cloud_frame_rotated, bytes_frame).result()
        new_cloud_list = bytes_cloud_frame_rotated(kwargs)
        list_buffer.extend(new_cloud_list)
        last_line_flag = data[44]
        if last_line_flag == b'\x80':
            # radar_stop 函数停止并关闭雷达连接，同时在RADAR_BUCKET中删除雷达id
            radar_stop(c_id=cid)
            # websocket对象的属性bucket中删除雷达id
            if cid in bucket:
                bucket.remove(cid)

    return


def bytes_cloud_frame_rotated(kwargs: dict):
    ws_id = kwargs['ws_id']
    # websocket = get_websocket_by_wsid(ws_id=ws_id)
    for i in manager.active_connections:
        if ws_id == i.clientId:
            logger.info("进入了正确的websocket判断区间")
            websocket = i
        else:
            websocket = manager.active_connections[0]

    logger.info(websocket)
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


async def start(ws_id):
    websocket = ctypes.cast(ws_id, ctypes.py_object).value
    for i in range(10):
        coal_yard = websocket.coalYard
        print('radars ==', coal_yard.coalRadarList)
        radars = coal_yard.coalRadarList
        for radar in radars:
            ip = radar.ip
            await websocket.send_text(data=str(ip))
            print('data == ', i)
            await asyncio.sleep(1.5)


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

        # 根据小点云文件(ndarray类型)计算体积和高度
        vom_start = datetime.now()
        vom_and_maxheight = await heap_vom_and_maxheight(cloud_ndarray=split_cloud_ndarray, minio_path=minio_path)
        vom_end = datetime.now()
        print(f"{heap.coalHeapName} 计算体积运行时间 === {vom_end - vom_start}")
        res.volume = vom_and_maxheight['volume']
        res.maxHeight = vom_and_maxheight['maxHeight']
        print("%s 体积: %.2f，高度: %.2f" % (res.coalHeapName, res.volume, res.maxHeight))

        list_cloud = split_cloud_ndarray.tolist()
        bytes_cloud = bytes(str(list_cloud), encoding='utf-8')
        # 上传文件至 minio,返回minio文件路径
        # 例如："http://172.16.200.243:9000/inventory-coal/2022/05/24/1_20220510144243A001.txt"
        print("list_cloud长度为" + str(len(list_cloud)))
        if len(list_cloud) != 0:
            data_buffer = io.BytesIO(bytes_cloud)
            res.cloudInfo = put_cloud_to_minio(f_name=minio_name, data=data_buffer, length=len(bytes_cloud))

        # 煤堆信息对象保存至 list
        res_list.append(res)
    return res_list


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app="main:application",
                host='0.0.0.0',
                port=8001,
                workers=1,
                reload=True)
