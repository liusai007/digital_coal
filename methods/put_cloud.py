import time
import minio
from config import settings


async def put_cloud(name, data, length):
    object_name = 'cloud_date/' + time.strftime("%Y/%m/%d/") + name
    # 例: inventory-coal/cloud_date/2022/06/14/a.txt
    minio_conf = settings.MINIO_CONF
    minio_client = minio.Minio(**minio_conf)
    minio_client.put_object(bucket_name='inventory-coal',
                            object_name=object_name,
                            # file_path=filepath,
                            data=data,
                            length=length)

    minio_path = "http://" + minio_conf['endpoint'] + '/inventory-coal/' + object_name
    print("minio_path == ", minio_path)
    return minio_path
