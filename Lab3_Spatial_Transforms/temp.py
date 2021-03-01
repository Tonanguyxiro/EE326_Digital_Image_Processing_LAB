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

if __name__ == '__main__':
    image1 = io.imread("Q3_3.tif")

    m,n = image1.shape

    number_of_pixel = m * n

    input_hist = []  # Distribution of input pixels
    for i in range(256):
        input_hist.append(np.sum(image1 == i))

    image2 = io
    output_hist = []  # Distribution of output pixels

