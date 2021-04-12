"""
LAB 3 Task II:

Specify a histogram for image Q3_2.tif.
"""

import numpy as np
from skimage import io, data
import math
import matplotlib.pyplot as plt

def sum(histogram, index):
    sum = 0
    for i in range(index):
        sum += histogram[i]
    # print(sum)
    return sum

def match(histogram, pixel):
    for i in range(histogram.shape[0]):
        if pixel < histogram[i]:
            return i

    return 0;

def spec_hist_1():
    spec_hist = np.zeros(256)

    spec_image = io.imread("Q3_2_spec.png")

    for i in range(256):
        spec_hist[i] = (np.sum(spec_image == i))

    spec_hist = spec_hist/np.sum(spec_hist)

    figure, ax = plt.subplots()
    ax.plot(np.range(256), spec_hist)
    plt.show()

    return spec_hist

def spec_hist_2():
    spec_hist = np.zeros(256)

    for i in range(20):
        spec_hist[i] = 0
    for i in range(20, 100):
        spec_hist[i] = 10*i
    for i in range(100, 210):
        spec_hist[i] = 2550 - 10*i
    for i in range(210, 256):
        spec_hist[i] = 256-i

    figure, ax = plt.subplots()
    ax.plot(range(256), spec_hist)
    plt.show()

    spec_hist = spec_hist/np.sum(spec_hist)
    return spec_hist

def spec_hist():
    spec_hist = np.zeros(256)
    for i in range(100):
        spec_hist[i] = 0
    for i in range(100, 210):
        spec_hist[i] = 25500 - 10*i
    for i in range(210, 256):
        spec_hist[i] = 256-i

    spec_hist = spec_hist/np.sum(spec_hist)

    figure, ax = plt.subplots()
    ax.plot(range(256), spec_hist)
    plt.show()

    return spec_hist

def hist_match_11810818(input_image, spec_hist):

# Define outputs
    output_image = np.zeros(input_image.shape, dtype=np.uint8)

    m,n = input_image.shape

    number_of_pixel = m * n

    input_hist = []  # Distribution of input pixels
    output_hist = []  # Distribution of output pixels

    # Count input
    for i in range(256):
        input_hist.append(np.sum(input_image == i))
    # print(input_hist)

    # histogram equalization
    for i in range(m):
        for j in range(n):
            output_image[i, j] = ((256-1)/number_of_pixel)*sum(input_hist, input_image[i, j])


    # histogram matching
    G_z = np.zeros((256), dtype=np.uint8)
    

    for i in range(256):
        # print((256-1)*sum(spec_hist, i))
        G_z[i] = (256-1)*sum(spec_hist, i)
    print(G_z)

    for i in range(m):
        for j in range(n): 
            output_image[i, j] = match(G_z, input_image[i, j])

    # Count output
    for i in range(256):
        output_hist.append(np.sum(output_image == i))


    return (output_image, output_hist, input_hist)


if __name__ == '__main__':

    
    [output_image_1, output_hist_1, input_hist_1] = hist_match_11810818(io.imread("Q3_2.tif"), spec_hist())
    
    # Print result
    io.imsave("Q3_2_11810818.tif", output_image_1)

    # Plot histogram
    fig1, [in_1, out_1] = plt.subplots(1, 2)
    in_1.plot(np.arange(256), input_hist_1)
    out_1.plot(np.arange(256), output_hist_1)

    plt.show()


