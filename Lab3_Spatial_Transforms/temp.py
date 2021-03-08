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
    image2 = io.imread("Q3_3_11810818.tif")


    m,n = image1.shape

    number_of_pixel = m * n

    input_hist = []  # Distribution of input pixels
    output_hist = []

    for i in range(256):
        input_hist.append(np.sum(image1 == i))
        output_hist.append(np.sum(image2 == i))


    fig1, [in_1, out_1] = plt.subplots(1, 2)
    in_1.plot(np.arange(256), input_hist)
    out_1.plot(np.arange(256), output_hist)



    plt.show()


