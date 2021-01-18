"""
LAB2 Task2

Use Python function “interp2” from packet “scipy” or your own written algorithm to interpolate a grey scale image by using bicubic interpolation.
Enlarge: [461, 461]
Shrink: [205, 205]
"""

import numpy as np
from skimage import io, data


def bicubic_11810818(input_file, dim):
    # Load image
    in_image = io.imread(input_file)
    out_width = dim[0]
    out_height = dim[1]
    in_width = in_image.shape[0]
    in_height = in_image.shape[1]
    out_image = np.zeros(dim, dtype=np.uint8)

    # Perform Exchange

    # Save Image
    io.imsave("bicubic_11810818.tif", out_image)


if __name__ == '__main__':
    bicubic_11810818("rice.tif", [461, 461])