import numpy as np
from skimage import io, data
import math
from scipy import interpolate

def interp_test():

    x = [1, 2, 3, 4]
    y = x;
    z = [[106, 110, 154, 188],
         [103,  99, 12, 157],
         [100,  97, 101, 109],
         [102, 104,  96, 105]]


    f1_1 = interpolate.interp1d(x, z[0], kind='cubic')
    f1_2 = interpolate.interp1d(x, z[1], kind='cubic')
    z1 = [f1_1(2.5), f1_2(2.5), f1_1(2.5), f1_2(2.5)]
    f1_3 = interpolate.interp1d(x, z1, kind='cubic')
    print("interpolate.interp1d get: " + str(f1_3(2.5)))

    f2 = interpolate.interp2d(x, y, z, kind='cubic')
    print("interpolate.interp2d get: " + str(f2(2.5, 2.5)[0]))

if __name__ == '__main__':
    interp_test()