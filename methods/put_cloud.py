import time
import minio
from config import settings


def put_cloud_to_minio(f_name, data, length):
    object_name = 'cloud_date/' + time.strftime("%Y/%m/%d/") + f_name
    # 例: inventory-coal/cloud_date/2022/06/14/a.txt
    minio_conf = settings.MINIO_CONF
    minio_client = minio.Minio(**minio_conf)
    minio_client.put_object(bucket_name='inventory-coal',
                            object_name=object_name,
                            data=data,
                            length=length)

    minio_path = "http://" + minio_conf['endpoint'] + '/inventory-coal/' + object_name
    print("minio_path == ", minio_path)
    return minio_path


def put_ply_to_minio(ply_name, file_path):
    object_name = 'cloud_date/' + time.strftime("%Y/%m/%d/") + ply_name
    # 例: inventory-coal/cloud_date/2022/06/14/a.txt
    minio_conf = settings.MINIO_CONF
    minio_client = minio.Minio(**minio_conf)

    try:
        minio_client.fput_object(bucket_name='inventory-coal',
                                 object_name=object_name,
                                 file_path=file_path)
    finally:
        minio_path = "http://" + minio_conf['endpoint'] + '/inventory-coal/' + object_name
        print("minio_path == ", minio_path)
        return minio_path
