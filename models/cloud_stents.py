from typing import List


class Stent(object):
    min_x: float
    max_x: float
    min_y: float
    max_y: float


stent_0 = object.__new__(Stent)
stent_0.min_x = 95.0
stent_0.max_x = 105.0
stent_0.min_y = 135.0
stent_0.max_y = 150.0

stent_1 = object.__new__(Stent)
stent_1.min_x = 120.0
stent_1.max_x = 135.0
stent_1.min_y = 135.0
stent_1.max_y = 150.0

stent_2 = object.__new__(Stent)
stent_2.min_x = 150.0
stent_2.max_x = 160.0
stent_2.min_y = 135.0
stent_2.max_y = 150.0

stent_3 = object.__new__(Stent)
stent_3.min_x = 175.0
stent_3.max_x = 195.0
stent_3.min_y = 135.0
stent_3.max_y = 150.0

stent_4 = object.__new__(Stent)
stent_4.X = [210.0, 220.0]
stent_4.Y = [135.0, 150.0]
stent_4.min_x = 210.0
stent_4.max_x = 220.0
stent_4.min_y = 135.0
stent_4.max_y = 150.0


stent_5 = object.__new__(Stent)
stent_5.min_x = 65.0
stent_5.max_x = 80.0
stent_5.min_y = 135.0
stent_5.max_y = 150.0

stent_6 = object.__new__(Stent)
stent_6.min_x = 40.0
stent_6.max_x = 50.0
stent_6.min_y = 135.0
stent_6.max_y = 150.0

stent_7 = object.__new__(Stent)
stent_7.min_x = 235.0
stent_7.max_x = 245.0
stent_7.min_y = 135.0
stent_7.max_y = 150.0

stents: List[Stent] = [stent_0, stent_1, stent_2, stent_3, stent_4, stent_5, stent_6, stent_7]
