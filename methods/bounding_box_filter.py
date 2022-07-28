import numpy
from methods.polygon_filter import isPoiWithinPoly


def bounding_box_filter(cloud_ndarray, heap_area):
    # 获得单个煤堆的 min_x,max_x,min_y,max_y
    # 获得多边形区域 poly_area=[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]]二维数组
    poly_area = []
    x_list = []
    y_list = []

    for p in heap_area:
        poly_area.append([p.x, p.y])
        x_list.append(p.x)
        y_list.append(p.y)

    # 多边形的起始点即终点
    origin_x = heap_area[0].x
    origin_y = heap_area[0].y
    poly_area.append([origin_x, origin_y])
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    split_cloud_ndarray = cloud_ndarray[(cloud_ndarray[:, 0] >= min_x) & (cloud_ndarray[:, 0] <= max_x)
                                        & (cloud_ndarray[:, 1] >= min_y) & (cloud_ndarray[:, 1] <= max_y)]
    # print("hahdsfha")
    return split_cloud_ndarray
    # 判断点位是否位于外矩形区域
    # x = cloud_ndarray[:, 0]
    # y = cloud_ndarray[:, 1]
    # z = cloud_ndarray[:, 2]

    # filename = save_url
    # f = open(filename, 'w')
    # print("创建成功:", filename)

    # for i in range(x.shape[0]):
    #     if (x[i] > float(min_x) and y[i] > float(min_y)
    #             and x[i] < float(max_x) and y[i] < float(max_y)):
    #         point = [x[i], y[i]]
    #         is_point_in_polygon = isPoiWithinPoly(point, poly_area)
    #         if is_point_in_polygon is True:
    #             f.write('%d %d %d\n' % (x[i], y[i], z[i]))

    # new_ndarray = numpy.genfromtxt(f)
    # return new_ndarray
