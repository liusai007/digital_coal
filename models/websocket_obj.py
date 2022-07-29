"""
@Time : 2022/7/13 17:03 
@Author : ls
@DES: 
"""
from fastapi import WebSocket
from models.custom_class import CoalRadar


class WebSocketObj():
    socket_id: int
    socket_obj: WebSocket
    args: list
