"""
@Time : 2022/5/26 10:44
@Author : lpy
@DES:
"""
import io
import ctypes
import asyncio
import requests
from methods.radar_func import *
from methods.volume_func import *
from methods.coal_yard import coal_yard_dict
from methods.send_data import send_frame_data
from methods.radar_func import radar_callback
from methods.put_cloud import put_cloud_to_minio
from methods.bounding_box_filter import bounding_box_filter
from methods.get_vom_and_maxheight import heap_vom_and_maxheight
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
        # print('客户端数量 ==', self.active_connections.__len__())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@application.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await websocket.send_text('Websocket已连接成功，正在等待用户指令！')
    ws_id = id(websocket)
    client_id = websocket.query_params['clientId']
    # 设置回调函数，将 websocket 内存地址作为参数，传入回调中
    set_callback_function(func=radar_callback, obj_id=ws_id)
    try:
        while True:
            data = await websocket.receive_text()
            yard_id = websocket.query_params['coalYardId']
            if data == 'start':
                await websocket.send_text("正在通过指令进行传输，请稍后...")
                base_url = settings.PATH_CONFIG.BASEURL
                url = base_url + '/coal/coalYard/realTime/coalYardInfo?coalYardId=' + str(yard_id)
                response = requests.get(url).json()
                # DictOBj将一个dict转化为一个对象，方便以属性的方式访问
                coal_yard_dict = response['data']
                coal_yard_obj = DictObj(coal_yard_dict)
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
                    await asyncio.sleep(2)
                    await websocket.send_text('开始盘煤')
                    # 设置空list用于在回调函数中保存雷达帧数据
                    websocket.listBuffer = list()
                    # websocket.__setattr__('listBuffer', list())
                    begin_response = radars_rotate_begin(radars, websocket)
                    if begin_response is False:
                        await websocket.send_text('存在未连接成功的雷达，操作中止......')
                        continue
                # await start(ws_id=ws_id)
            else:
                await websocket.send_text(data='输入的指令暂时无法识别，请联系管理员')
                continue

            # 发送长度为3000的数据帧，n代表一次发送的数据长度
            result = await send_frame_data(websocket, n=3000)
            if result == "success":
                cloud_list = websocket.listBuffer
                cloud_ndarray = numpy.array(cloud_list)
                # 实现用于切割煤场并计算体积的功能
                res_list = await split_and_calculate_volume(coal_yard=coal_yard_obj, cloud_ndarray=cloud_ndarray)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")


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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app="main:application",
                host='127.0.0.1',
                port=8000,
                workers=1,
                reload=True)
