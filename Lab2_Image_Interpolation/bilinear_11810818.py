"""
LAB2 Task 1_2

Use bilinear interpolation interpolation to interpolate a grey scale image
Enlarge: [461, 461]
Shrink: [205, 205]
"""

import numpy as np
from skimage import io, data

def bilinear_11810818(input_file, dim):
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
            x = col*((in_width-1)/(out_width-1))
            y = row*((in_height-1)/(out_height-1))
            if x == round(x):
                if y == round(y):
                    out_image[col, row] = in_image[x, y]
                else:
                    out_image[col, row] =

            elif y == round(y):

            else:


    # Save Image
    io.imsave("bilinear_11810818.tif", out_image)


if __name__ == '__main__':
    bilinear_11810818("rice.tif", [461, 461])