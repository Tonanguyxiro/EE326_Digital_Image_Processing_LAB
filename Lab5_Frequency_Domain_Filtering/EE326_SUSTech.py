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
    output_image = output_image.astype(np.uint8)

    return output_image

# LAB 4

def convolution_3x3(input_image, operator_3x3):
    col, row = input_image.shape
    output_image = np.zeros([col, row])

    operator = np.zeros(9)
    for i in range(3):
        for j in range(3):
            operator[i*3+j] = operator_3x3[i, j]

    for i in range(col):
        for j in range(row):
            local = np.zeros(9)
            index = 0
            for i2 in [-1, 0, 1]:
                for j2 in [-1, 0, 1]:
                    if (i + i2 < 0 or i + i2 > col - 1 or j + j2 < 0 or j + j2 > row - 1):
                        local[index] = 0
                    else:
                        local[index] = input_image[i + i2, j + j2]
                    index += 1

            output_image[i, j] = np.dot(operator, local)

    output_image = output_image.astype(np.uint8)

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

def extract_result(input_image):
    x, y = input_image.shape
    output_image = input_image[int(x/2):x, int(y/2):y]

    return output_image


def zero_padding_DFT(input_image, P, Q):
    m,n = input_image.shape

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


def transform_centering(input_image):
    m, n = input_image.shape
    output_image = input_image
    mul1 = 1
    for i in range(m):
        mul2 = mul1
        for j in range(n):
            output_image[i, j] = input_image[i, j] * mul2
            mul2 *= -1
        mul1 *= -1
    return output_image


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


def butterworth_filter(b, a, n, sigma):
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    print(x)
    x = x - a / 2
    y = y - b / 2
    d = np.sqrt(x * x + y * y)
    h = 1/(1+(d/sigma)**(2*n))
    return h


if __name__ == '__main__':

    io.imsave("butterworth_filter_test.tif", format_image(butterworth_filter(100, 100, 2, 30)))

