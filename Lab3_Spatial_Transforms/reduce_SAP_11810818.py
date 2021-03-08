"""
LAB 3 Task IV:

"""

import numpy as np
from skimage import io, data
import math
import matplotlib.pyplot as plt

def reduce_SAP_11810818(input_image, n_size):

    output_image = np.zeros(input_image.shape, dtype=np.uint8)

    m,n = input_image.shape
    number_of_pixel = m * n

    for i in range(m):
        for j in range(n):
            step = (int)((n_size-1)/2)
            pixels = np.zeros(n_size*n_size) 
            
            for i2 in range(n_size):
                for j2 in range(n_size):
                    if i-step+i2 >= 0 and i-step+i2 < input_image.shape[0] and j-step+j2 >= 0 and j-step+j2 < input_image.shape[0]:
                        pixels[j2*n_size+i2] = input_image[i-step+i2, j-step+j2]

            # print(pixels)
            pixels = np.sort(pixels, axis=None)
            # print(pixels)

            output_image[i, j] = pixels[(int)((n_size*n_size-1)/2)]


    return output_image

if __name__ == '__main__':

    output_image_1 = reduce_SAP_11810818(io.imread("Q3_4.tif"), 3)
    
    # Print result
    io.imsave("Q3_4_11810818.tif", output_image_1)


