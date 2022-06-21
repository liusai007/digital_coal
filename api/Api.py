"""
@Time : 2022/5/26 11:50 
@Author : lpy
@DES: api路由
"""
from fastapi import APIRouter
from api.endpoints import coal, radar, radar_test

api_router = APIRouter(prefix="/api")
api_router.include_router(coal.router, prefix='/coal', tags=["盘煤管理"])
# api_router.include_router(radar.router, prefix='/radar', tags=["雷达管理"])
api_router.include_router(radar_test.router, prefix='/radar', tags=["雷达测试"])

