# highboost_filtering_11810818.py

import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt

def unsharp_masking_11810818(input_image):
    m, n = input_image.shape
    output_image = np.zeros([m,n])

    operator = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1])

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
                    
            output_image[i, j] = np.dot(operator, local)

    output_image = np.clip(output_image, 0, 255)

    output_image = 1.5*input_image + output_image

    output_image = np.clip(output_image, 0, 255)

    output_image = output_image.astype(np.uint8)


    
    return output_image

if __name__ == '__main__':
# Image 1

    # Process image
    output_image_1 = unsharp_masking_11810818(io.imread("Q4_1.tif"))
    
    # Print result
    io.imsave("Q4_1_3_11810818.tif", output_image_1)

# Image 1

    # Process image
    output_image_1 = unsharp_masking_11810818(io.imread("Q4_2.tif"))
    
    # Print result
    io.imsave("Q4_2_3_11810818.tif", output_image_1)