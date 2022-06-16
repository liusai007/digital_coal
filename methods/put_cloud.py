import time
import minio
from config import settings


def put_cloud(filepath, filename):
    object_name = 'cloud_date/' + time.strftime("%Y/%m/%d/") + filename
    # inventory-coal/cloud_date/2022/06/14/a.txt
    minio_conf = settings.MINIO_CONF
    minio_client = minio.Minio(**minio_conf)
    minio_client.fput_object(bucket_name='inventory-coal',
                             object_name=object_name,
                             file_path=filepath,
                             content_type="application/csv")

    minio_path = "http://" + minio_conf['endpoint'] + '/inventory-coal/' + object_name
    print("minio_path == ", minio_path)
    return minio_path