"""
@Time : 2022/7/28 14:25 
@Author : ls
@DES:
"""


# 每次读取一个长度为 n 的list,最终读取全部list
async def send_frame_data(websocket, n: int):
    i = 1
    while True:
        # if websocket.conn_status == 'off':
        conn_bucket = websocket.conn_radarsBucket
        if conn_bucket.__len__() == 0:
            # 代表雷达停止的条件
            await websocket.send_text('===========数据发送结束===========')
            break
        elif len(websocket.list_buffer) >= n * i:
            ws_list = websocket.list_buffer
            first_list = ws_list[0: n * i]
            # print('first_list == ', first_list)
            result = await websocket.send_text(str(first_list))
            # print('result ==', result)
            i += 1

            while True:
                if len(websocket.list_buffer) >= n * i:
                    ws_list = websocket.list_buffer
                    loop_list = ws_list[(i - 1) * n:i * n]
                    # print('loop_list == ', loop_list)
                    result = await websocket.send_text(str(loop_list))
                    # print('result ==', result)
                    # await asyncio.sleep(0.1)
                    i += 1
                    if len(websocket.list_buffer) < n * i:
                        websocket.conn_radarsBucket.clear()
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
                                print('last_list == ', last_list)
                                await websocket.send_text(str(last_list))
                                await websocket.send_text("数据发送完成！")
                            websocket.conn_radarsBucket.clear()
                            break
        else:
            continue

    return "success"
