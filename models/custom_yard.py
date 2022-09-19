"""
@Time : 2022/9/8 8:52 
@Author : ls
@DES: 
"""
from models.custom_stent import stents

COAL_YARDS = {
    11: {
        'split_list': [[10.5, 12.5, 8], [12.5, 14.0, 10], [14.0, 15.0, 11], [15.0, 16.0, 11], [16.0, 17.0, 13],
                       [17.0, 18.0, 14], [18.0, 19.0, 16], [19.0, 20.0, 18], [20.0, 21.0, 20], [21.0, 180.0, 26],
                       [180.0, 192.0, 10.0], [192.0, 195.0, 8.0], [195.0, 198.0, 6.0], [198.0, 200.0, 5.0],
                       [200.0, 202.0, 4.0], [202.0, 204.0, 2.0], [204.0, 208.0, 2.0], [208.0, 212.0, 1.0],
                       [212.0, 216.0, 1.0], [216.0, 220.0, 1.0], [220.0, 240.0, 1.0]],
        'polygon': [[8.0, 8.0], [420.0, 8.0], [420.0, 170.0], [8.0, 220.0], [8.0, 8.0]],
        'stents': stents
    },
    9: {
        'split_list': [[10.5, 12.5, 8], [12.5, 14.0, 10], [14.0, 15.0, 11], [15.0, 16.0, 11], [16.0, 17.0, 13],
                       [17.0, 18.0, 14], [18.0, 19.0, 16], [19.0, 20.0, 18], [20.0, 21.0, 20], [21.0, 180.0, 26],
                       [180.0, 192.0, 10.0], [192.0, 195.0, 8.0], [195.0, 198.0, 6.0], [198.0, 200.0, 5.0],
                       [200.0, 202.0, 4.0], [202.0, 204.0, 2.0], [204.0, 208.0, 2.0], [208.0, 212.0, 1.0],
                       [212.0, 216.0, 1.0], [216.0, 220.0, 1.0], [220.0, 240.0, 1.0]],
        'polygon': [[8.0, 8.0], [420.0, 8.0], [420.0, 170.0], [8.0, 220.0], [8.0, 8.0]],
        'stents': stents
    }
}
