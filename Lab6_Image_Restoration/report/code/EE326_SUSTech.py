'''
Library USE for EE326 2021
'''

import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt


# General
def format_image(input_image):
    output_image = input_image
    output_image -= np.min(output_image)
    output_image = (output_image/np.max(output_image))*255
    return output_image

# LAB 4


def convolution_3x3(input_image, operator_3x3):
    col, row = input_image.shape
    output_image = np.zeros([col, row])
    input_image = np.pad(input_image, 1)
    for i in range(0, col):
        for j in range(0, row):
            for i2 in range(3):
                for j2 in range(3):
                    output_image[i, j] += input_image[i+i2, j+j2] * operator_3x3[i2, j2]

    return output_image


def sobel_filter(input_image):

    operator1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    operator2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    output_image1 = convolution_3x3(input_image, operator1)
    output_image2 = convolution_3x3(input_image, operator2)
    #
    # output_image1 = np.clip(output_image1, 0, 255)
    # output_image2 = np.clip(output_image2, 0, 255)

    output_image = output_image1 + output_image2 # + input_image
    # output_image = np.clip(output_image, 0, 255)

    output_image = output_image.astype(np.uint8)

    return output_image


def zero_padding(input_image, P, Q):
    output_image = np.zeros([P, Q])

    return output_image

# LAB 5


def extract_result_eastsouth(input_image):
    x, y = input_image.shape
    output_image = input_image[int(x/2):x, int(y/2):y]

    return output_image


def extract_result_westnorth(input_image):
    x, y = input_image.shape
    output_image = input_image[0:int(x/2), 0:int(y/2)]

    return output_image


def zero_padding_DFT(input_image, P, Q):
    m, n = input_image.shape

    output_image = np.zeros([P, Q])
    output_image[0:m, 0:n] = input_image

    return output_image


def zero_padding_DFT(input_image):
    m,n = input_image.shape

    P = 2*m
    Q = 2*n

    output_image = np.zeros([P, Q])
    output_image[0:m, 0:n] = input_image

    return output_image


def centering(size):
    m, n = size
    centering_matrix = np.ones(size)
    mul1 = 1
    for i in range(m):
        mul2 = mul1
        for j in range(n):
            centering_matrix[i, j] = centering_matrix[i, j] * mul2
            mul2 *= -1
        mul1 *= -1
    return centering_matrix


def generating_from_spatial_filter(input_filter, P, Q):
    output_filter = np.zeros(P, Q)

    return output_filter


def gaussian_filter(a, b, sigma):
    x, y = np.meshgrid(np.linspace(0, a-1, a), np.linspace(0, b-1, b))
    x = x - a/2
    y = y - b/2
    d = x * x + y * y
    g = np.exp(-(d / (2.0 * sigma ** 2)))
    # g = g/np.sum(g)
    return g


def butterworth_filter(b, a, center, n, sigma):
    cx, cy = center
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    x = x - cx
    y = y - cy
    d = np.sqrt(x * x + y * y)
    h = 1/((1+(d/sigma))**(2*n))
    return h

if __name__ == '__main__':

    test_input = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    print(transform_centering(test_input))

