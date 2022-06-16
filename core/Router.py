"""
@Time : 2022/5/26 12:34 
@Author : lpy
@DES: 路由聚合
"""
from fastapi import APIRouter
from api.Api import api_router
# from views.views import views_router

router = APIRouter()
# API路由
router.include_router(api_router);
# 视图路由
# router.include_router(views_router)
