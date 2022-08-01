"""
@Time : 2022/5/26 11:50 
@Author : lpy
@DES: api路由
"""
from fastapi import APIRouter
from api.endpoints import coal, coal_test, coal_test_v1, coal_test_v2

api_router = APIRouter(prefix="/api")
api_router.include_router(coal.router, prefix='/coal', tags=["盘煤测试"])
# api_router.include_router(coal_test.router, prefix='/coal_test', tags=["煤场测试"])
# api_router.include_router(coal_test_v1.router, prefix='/coal_test_v1', tags=["煤场测试_v1"])
api_router.include_router(coal_test_v2.router, prefix='/coal', tags=["煤场测试-标准版"])
