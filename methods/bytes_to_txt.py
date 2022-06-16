import numpy as np


# filename .cld 二进制文件，单位是 厘米
def write_bytes_to_txt(filename):
    '''
    :param filename: 与二进制 .cld 同一个路径
    :return: 返回 txt 文件路径
    '''
    txt_cloud_path = filename.replace('.cld', '.txt')
    with open(filename, mode='rb') as f:
        # mode='rb' 读取二进制文件模式
        bytes_data = f.read()
        # 二进制buffer 转为 ndarray
        data = np.frombuffer(bytes_data, np.int16)
        # print('len(data):', len(data))
        x = int(len(bytes_data) / 6)
        # 改变 ndarray 的形状
        reshape_data = data.reshape(x, 3)
        # ndarray 保存为 txt 文件，单位是 厘米
        np.savetxt(txt_cloud_path, reshape_data, fmt='%d')

    return txt_cloud_path


