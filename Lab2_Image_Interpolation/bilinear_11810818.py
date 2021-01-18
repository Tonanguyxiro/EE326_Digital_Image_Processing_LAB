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


def bilinear_11810818(input_file, dim):
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
            x = col*((in_width-1)/(out_width-1))
            y = row*((in_height-1)/(out_height-1))
            if x == round(x):
                if y == round(y):  # 在点上
                    out_image[col, row] = in_image[round(x), round(y)]
                else:  # 在纵轴上
                    out_image[col, row] = round(linear(y-math.floor(y), in_image[round(x), math.floor(y)], in_image[round(x), math.floor(y)+1]))
            elif y == round(y):  # 在横轴上
                out_image[col, row] = round(linear(x-math.floor(x), in_image[math.floor(x), round(y)], in_image[math.floor(x)+1, round(y)]))
            else:
                left=linear(y-math.floor(y), in_image[math.floor(x), math.floor(y)], in_image[math.floor(x), math.floor(y)+1])
                right=linear(y-math.floor(y), in_image[math.floor(x)+1, math.floor(y)], in_image[math.floor(x)+1, math.floor(y)+1])
                out_image[col, row] = round(linear(x-math.floor(x), right, left))
                # out_image[col, row] = 1

    print(out_image)
    # Save Image
    io.imsave("bilinear_11810818.tif", out_image)


if __name__ == '__main__':
    bilinear_11810818("rice.tif", [461, 461])