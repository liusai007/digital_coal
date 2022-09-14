"""
@Time : 2022/7/28 14:25 
@Author : ls
@DES:
"""

# 每次读取一个长度为 n 的list,最终读取全部list
import numpy
from scipy.spatial import Delaunay


def cloud_data_generator(websocket, n: int):
    i = 1
    while True:
        conn_bucket = websocket.conn_radarsBucket
        if conn_bucket.__len__() == 0:
            # 代表雷达停止的条件
            break
        elif len(websocket.list_buffer) >= n * i:
            ws_list = websocket.list_buffer
            first_list = ws_list[0: n * i]
            yield first_list
            i += 1

            while True:
                if len(websocket.list_buffer) >= n * i:
                    ws_list = websocket.list_buffer
                    loop_list = ws_list[(i - 1) * n:i * n]
                    yield loop_list
                    i += 1
                else:
                    # 如果websocket对象中的雷达连接list不是空值，返回继续循环
                    if len(websocket.conn_radarsBucket) != 0:
                        continue
                    # 如果雷达连接list是空值，则发送可能遗漏的数据帧
                    else:
                        if len(websocket.list_buffer) >= n * i:
                            break
                        elif len(websocket.list_buffer) < n * i:
                            ws_list = websocket.list_buffer
                            last_list = ws_list[n * (i - 1):]
                            if last_list.__len__() != 0:
                                yield last_list
                            break
        else:
            continue

    yield "success"


# 每次读取一个长度为 n 的list,最终读取全部list
async def send_cloud_data(websocket, n: int):
    i = 1
    while True:
        conn_bucket = websocket.conn_radarsBucket
        if conn_bucket.__len__() == 0:
            # 代表雷达停止的条件
            await websocket.send_text('数据发送结束')
            break
        elif len(websocket.list_buffer) >= n * i:
            ws_list = websocket.list_buffer
            first_list = ws_list[0: n * i]
            await websocket.send_text(str(first_list))
            i += 1

            while True:
                if len(websocket.list_buffer) >= n * i:
                    ws_list = websocket.list_buffer
                    loop_list = ws_list[(i - 1) * n:i * n]
                    await websocket.send_text(str(loop_list))
                    i += 1
                else:
                    # 如果websocket对象中的雷达连接list不是空值，返回继续循环
                    if len(websocket.conn_radarsBucket) != 0:
                        continue
                    # 如果雷达连接list是空值，则发送可能遗漏的数据帧
                    else:
                        if len(websocket.list_buffer) >= n * i:
                            break
                        elif len(websocket.list_buffer) < n * i:
                            ws_list = websocket.list_buffer
                            last_list = ws_list[n * (i - 1):]
                            if last_list.__len__() != 0:
                                await websocket.send_text(str(last_list))
                            break
        else:
            continue

    return 'success'


async def send_mesh_data(cloud: numpy.ndarray, websocket):
    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]

    tri = Delaunay(numpy.array([x, y]).T)
    for i in range(x.shape[0]):
        data = ('%.2f %.2f %.2f\n' % (x[i], y[i], z[i]))
        await websocket.send_text(data)
    for vert in tri.simplices:
        data = ('3 %d %d %d\n' % (vert[0], vert[1], vert[2]))
        await websocket.send_text(data)


