"""
LAB2 Task 1_2

Use bilinear interpolation interpolation to interpolate a grey scale image
Enlarge: [461, 461]
Shrink: [205, 205]
"""

import numpy as np
from skimage import io, data
import math


def linear(x, y1, y2):
    if y2 > y1:
        return y1 + x * (y2-y1)
    else:
        return y2 + (1-x)*(y1-y2)


def small_map(x, range):
    ratio = x/range
    return 0.01 + ratio*(range-0.02)


def bilinear3_11810818(input_file, dim):
    # Load image
    in_image = io.imread(input_file)
    print(in_image)
    out_width = dim[0]
    out_height = dim[1]
    in_width = in_image.shape[0]
    in_height = in_image.shape[1]
    out_image = np.zeros(dim, dtype=np.uint8)
    # out_image = np.zeros(dim)

    # Perform Exchange
    for col in range(out_width):
        for row in range(out_height):
            x = small_map(col*((in_width-1)/(out_width-1)), out_width)
            y = small_map(row*((in_height-1)/(out_height-1)), out_height)

            left=linear(y-math.floor(y), in_image[math.floor(x), math.floor(y)], in_image[math.floor(x), math.floor(y)+1])
            right=linear(y-math.floor(y), in_image[math.floor(x)+1, math.floor(y)], in_image[math.floor(x)+1, math.floor(y)+1])

            out_image[col, row] = round(linear(x-math.floor(x), left, right))

    # Save Image
    io.imsave("shrank_bilinear_11810818.tif", out_image)


if __name__ == '__main__':
    bilinear3_11810818("rice.tif", [205, 205])