"""
@Time : 2022/6/16 11:48 
@Author : ls
@DES: 
"""
from typing import List
from models.custom_class import CoalRadar
from api.endpoints.radar import radar_start
from fastapi import APIRouter

router = APIRouter()


@router.post("/radar_test", summary="雷达测试")
async def radar_test(radars: List[CoalRadar]):
    radar_start(radars)



