"""
@Time : 2022/7/11 13:45 
@Author : ls
@DES: 
"""


class DictObj(object):
    # 嵌套的字典(dict)转换成object对象，可以方便直接访问对象的属性的方法
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [DictObj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, DictObj(b) if isinstance(b, dict) else b)
