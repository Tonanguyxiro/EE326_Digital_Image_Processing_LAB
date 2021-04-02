"""
LAB 3 Task I:

Implement the histogram equalization to the input images Q3_1_1.tif and Q3_1_2.tif.
"""

import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt


def sum(histogram, index):
    sum = 0
    for i in range(index):
        sum = sum + histogram[i]
    # print(sum)
    return sum

def hist_equ_11810818(input_image):

    # Define outputs
    output_image = np.zeros(input_image.shape, dtype=np.uint8)

    m,n = input_image.shape

    number_of_pixel = m * n

    input_hist = []  # Distribution of input pixels
    output_hist = []  # Distribution of output pixels

    # Count input
    for i in range(256):
        input_hist.append(np.sum(input_image == i)/number_of_pixel)
    # print(input_hist)

    # histogram equalization
    for i in range(m):
        for j in range(n):
            output_image[i, j] = ((256-1))*sum(input_hist, input_image[i, j])

    # Count output
    for i in range(256):
        output_hist.append(np.sum(output_image == i)/number_of_pixel)
    

    return (output_image, output_hist, input_hist)

if __name__ == '__main__':
# Image 1

    # Process image
    [output_image_1, output_hist_1, input_hist_1] = hist_equ_11810818(io.imread("Q3_1_1.tif"))
    
    # Print result
    io.imsave("Q3_1_1_11810818.tif", output_image_1)

    # Plot histogram
    fig1, [in_1, out_1] = plt.subplots(1, 2)
    in_1.plot(np.arange(256), input_hist_1)
    out_1.plot(np.arange(256), output_hist_1)

# Image 2

    # Process image
    [output_image_2, output_hist_2, input_hist_2] = hist_equ_11810818(io.imread("Q3_1_2.tif"))
    
    # Print result
    io.imsave("Q3_1_2_11810818.tif", output_image_2)

    # Plot histogram
    fig2, [in_2, out_2] = plt.subplots(1, 2)
    in_2.plot(np.arange(256), input_hist_2)
    out_2.plot(np.arange(256), output_hist_2)

    plt.show()

    




