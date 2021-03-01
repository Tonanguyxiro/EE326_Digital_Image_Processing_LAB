"""
LAB 3 Task III:

Implement the local histogram equalization to the input images Q3_3.tif.
"""

import numpy as np
from skimage import io, data
import math
import matplotlib.pyplot as plt

def extract_lacal(input_image, x, y, m_size):
    step = int((m_size-1)/2)
    local = np.zeros((m_size, m_size), dtype=np.uint8)

    for i in range(x - step, x + step):
        for j in range(y - step, y + step):
            if i >= 0 and i < input_image.shape[0] and j >= 0 and j < input_image.shape[0]:
                local[i - (x - step), j - (y - step)] = input_image[i, j]

    return local

def hist_equ(local):

    number_of_pixel = local.shape[0]*local.shape[1]

    center_x = int((local.shape[0]-1)/2)
    center_y = int((local.shape[1]-1)/2)

    input_hist = []  # Distribution of input pixels
    # Count input
    for i in range(256):
        input_hist.append(np.sum(local == i))

    output = ((256-1)/number_of_pixel)*sum(input_hist, local[center_x, center_y])

    return output

def sum(histogram, index):
    sum = 0
    for i in range(index):
        sum = sum + histogram[i]
    # print(sum)
    return sum

def local_hist_equ_11810818(input_image, m_size):

    output_image = np.zeros(input_image.shape, dtype=np.uint8)

    m,n = input_image.shape
    number_of_pixel = m * n
    
    input_hist = []  # Distribution of input pixels
    output_hist = []  # Distribution of output pixels

    # Count input
    for i in range(256):
        input_hist.append(np.sum(input_image == i))
    # print(input_hist)

    # local histogram equalization
    for i in range(m):
        for j in range(n):
            print("(" + str(i)+", "+str(j)+")")
            local = extract_lacal(input_image, i, j, m_size)
            output_image[i, j] = hist_equ(local)

    # Count output
    for i in range(256):
        output_hist.append(np.sum(output_image == i))


# Insert code here 
    return (output_image, output_hist, input_hist)


if __name__ == '__main__':

    [output_image_1, output_hist_1, input_hist_1] = local_hist_equ_11810818(io.imread("Q3_3.tif"), 3)
    
    # Print result
    io.imsave("Q3_3_11810818.tif", output_image_1)

    # Plot histogram
    fig1, [in_1, out_1] = plt.subplots(1, 2)
    in_1.plot(np.arange(256), input_hist_1)
    out_1.plot(np.arange(256), output_hist_1)




