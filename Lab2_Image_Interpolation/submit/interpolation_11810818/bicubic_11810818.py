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
    ratio = (x-1)/(range)
    return 2.01 + ratio*(range-4.02)


def find_value(x, y, z, pix_x, pix_y):
    f = interpolate.interp2d(x, y, z, kind='cubic')
    out = f(pix_x, pix_y)[0]
    print(out)
    print(z)

    return out


def bicubic2_11810818(input_file, dim):
    # Load image
    in_image = io.imread(input_file)
    out_width = dim[0]
    out_height = dim[1]
    in_width = in_image.shape[0]
    in_height = in_image.shape[1]
    out_image = np.zeros(dim, dtype=np.uint8)

    print(in_image)

    # Perform Exchange
    for col in range(out_width):
        for row in range(out_height):
            pix_x = small_map2(col*((in_width-1)/(out_width-1)), out_width)
            pix_y = small_map2(row*((in_height-1)/(out_height-1)), out_height)
            space = np.zeros([4, 4])
            x_range = [math.floor(pix_x) - 1, math.floor(pix_x), math.floor(pix_x) + 1, math.floor(pix_x) + 2]
            y_range = [math.floor(pix_y) - 1, math.floor(pix_y), math.floor(pix_y) + 1, math.floor(pix_y) + 2]

            # print("x: "+str(x_range))
            # print("y: "+str(y_range))
            # print(str(pix_x) + " " + str(pix_y))

            for i in range(4):
                for j in range(4):
                    space[i, j] = in_image[x_range[i], y_range[j]]

            out_image[col, row] = find_value(
                x_range,
                y_range,
                space,
                pix_x,
                pix_y
            )

    # Save Image
    print(out_image)
    io.imsave("enlarged_bicubic3_11810818.tif", out_image)


if __name__ == '__main__':
    # bicubic2_11810818("rice.tif", [205, 205])
    bicubic2_11810818("rice.tif", [461, 461])

