"""
LAB2 Task2

Use Python function “interp2” from packet “scipy” or your own written algorithm to interpolate a grey scale image by using bicubic interpolation.
Enlarge: [461, 461]
Shrink: [205, 205]
"""

import numpy as np
from skimage import io, data
import math
from scipy import interpolate

def small_map2(x, range):
    ratio = x/range
    return 3.01 + ratio*(range-8.02)

def bicubic_11810818(input_file, dim):
    # Load image
    in_image = io.imread(input_file)
    out_width = dim[0]
    out_height = dim[1]
    in_width = in_image.shape[0]
    in_height = in_image.shape[1]
    out_image = np.zeros(dim, dtype=np.uint8)

    # Perform Exchange
    for col in range(out_width):
        for row in range(out_height):
            pix_x = small_map2(col*((in_width-1)/(out_width-1)), out_width)
            pix_y = small_map2(row*((in_height-1)/(out_height-1)), out_height)
            x=[math.floor(pix_x)-1, math.floor(pix_x), math.floor(pix_x)+1, math.floor(pix_x)+2]
            y=[math.floor(pix_y)-1, math.floor(pix_y), math.floor(pix_y)+1, math.floor(pix_y)+2]
            z=in_image[math.floor(pix_x)-1:math.floor(pix_x)+3, math.floor(pix_y)-1:math.floor(pix_y)+3]
            f = interpolate.interp2d(x, y, z, kind='cubic')

            out_image[col, row] = f(col, row)

    # Save Image
    print(out_image)
    io.imsave("bicubic_11810818.tif", out_image)


if __name__ == '__main__':
    bicubic_11810818("rice.tif", [461, 461])