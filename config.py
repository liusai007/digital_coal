"""
@Time : 2022/5/26 10:44
@Author : lpy
@DES: 配置文件
"""

import os.path
from pydantic import BaseSettings
from typing import List


class PathConfig(object):
    BASEURL: str = "http://172.16.200.239:9070/stage-api"


class RadarConfig():
    RUNMODE: int = 0
    SPEED: int = 64
    ANGLESCENESCAN: int = 360


class Config(BaseSettings):
    # 调试模式
    APP_DEBUG: bool = True
    # 项目信息
    PROJECT_NAME: str = "激光盘煤仪项目"
    DESCRIPTION: str = "盘煤仪接口文档"
    VERSION: str = "1.0.0'"
    # 静态资源目录
    STATIC_DIR: str = os.path.join(os.getcwd(), "static")
    TEMPLATE_DIR: str = os.path.join(STATIC_DIR, "views")
    # 跨域请求
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    # Session
    SECRET_KEY = "session"
    SESSION_COOKIE = "session_id"
    SESSION_MAX_AGE = 14 * 24 * 60 * 60
    CONNECT_ERROR_TIME = 10
    # minio 配置
    MINIO_CONF = {
        'endpoint': '172.16.200.243:9000',
        'access_key': 'minioadmin',
        'secret_key': 'minioadmin',
        'secure': False
    }
    # 雷达参数配置
    RADAR = RadarConfig()
    PATH_CONFIG = PathConfig()
    DATA_PATH = '/opt/python/coal_data'
    CLOUD_COMBINED_PATH = DATA_PATH + '/combined_cloud'
    CLOUD_SAMPLED_PATH = DATA_PATH + '/sampled_cloud'


settings = Config()
