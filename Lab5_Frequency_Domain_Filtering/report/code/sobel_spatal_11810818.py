# gradient method

# unsharp_masking_11810818.py

import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import EE326_SUSTech

def unsharp_masking_11810818(input_image):
    m, n = input_image.shape
    output_image = np.zeros([m,n])
    output_image1 = np.zeros([m,n])
    output_image2 = np.zeros([m,n])

    operator1 = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
    operator2 = np.array([-1, 0, 1, -2, 0, -2, -1, 0, 1])
    


    for i in range(m):
        for j in range(n):
            local = np.zeros(9)
            index = 0
            for i2 in [-1, 0, 1]:
                for j2 in [-1, 0, 1]:
                    if (i + i2 < 0 or i + i2 > m-1 or j + j2 < 0 or j + j2 > n-1):
                        local[index] = 0
                    else:
                        local[index] = input_image[i+i2, j+j2]
                    index += 1
                    
            output_image1[i, j] = np.dot(operator1, local)
            output_image2[i, j] = np.dot(operator2, local)

    output_image1 = np.clip(output_image1, 0, 255)
    output_image2 = np.clip(output_image2, 0, 255)

    output_image = output_image1 + output_image2 \
                   #+ input_image
    # output_image = np.clip(output_image, 0, 255)

    # output_image = output_image.astype(np.uint8)

    output_image = EE326_SUSTech.format_image(output_image)
    io.imsave("Q5_1_4_sobel_spatial.tif", output_image)
    return output_image

if __name__ == '__main__':
# Image 1

    # Process image
    unsharp_masking_11810818(io.imread("Q5_1.tif"))
    
    # Print result


# # Image 1
#
#     # Process image
#     output_image_1 = unsharp_masking_11810818(io.imread("Q4_2.tif"))
#
#     # Print result
#     io.imsave("Q4_2_4_11810818.tif", output_image_1)