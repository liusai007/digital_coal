import numpy
from methods.polygon_filter import is_poi_within_polygon


def remove_cover_by_list(cloud: numpy.ndarray, s_list, polygon):
    new_list = []
    for s in s_list:
        # s: [x, y, z]
        split_cloud = cloud[(cloud[:, 1] >= s[0]) & (cloud[:, 1] < s[1]) & (cloud[:, 2] <= s[2])]
        if s[1] >= 200.0:
            split_cloud = remove_out_polygon_point(cloud=split_cloud, polygon=polygon)
        new_list.extend(split_cloud)

    new_cloud = numpy.array(new_list)
    return new_cloud


def remove_out_polygon_point(cloud: numpy.ndarray, polygon):
    new_list = []
    rows = cloud.shape[0]

    for row in range(rows):
        point = cloud[row]
        p_list = point[0:2].tolist()
        if is_poi_within_polygon(p_list, polygon):
            new_list.append(point)

    new_array = numpy.array(new_list)
    return new_array
