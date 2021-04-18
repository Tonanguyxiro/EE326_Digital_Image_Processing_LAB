'''
Library USE for EE326 2021
'''

import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from numba import njit,prange

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


def denoise_filter(input_image, n_size, mode):
    output_image = np.zeros(input_image.shape, dtype=np.uint8)

    m, n = input_image.shape

    for i in range(m):
        for j in range(n):
            step = (int)((n_size - 1) / 2)
            pixels = np.zeros(n_size * n_size)

            for i2 in range(n_size):
                for j2 in range(n_size):
                    if i - step + i2 >= 0 \
                            and i - step + i2 < input_image.shape[0] \
                            and j - step + j2 >= 0 \
                            and j - step + j2 < input_image.shape[0]:
                        pixels[j2 * n_size + i2] = input_image[i - step + i2, j - step + j2]

            pixels = np.sort(pixels)

            if(mode == "max"):
                output_image[i, j] = pixels[pixels.shape[0]-1]
            elif(mode == "medium"):
                output_image[i, j] = pixels[(int)((n_size * n_size + 1) / 2)]
            elif(mode == "min"):
                output_image[i, j] = pixels[0]
            elif(mode == "average"):
                output_image[i, j] = np.average(pixels)
            elif(mode == "smart"):
                have_normal_pixel = 0
                for pixel in pixels:
                    if(pixel < 250 and pixel > 5):
                        have_normal_pixel = 1
                if(have_normal_pixel):
                    selected = (int)(pixels.shape[0]/2)
                    while(pixels[selected] < 5):
                        selected = selected + 1
                    while(pixels[selected] > 250):
                        selected = selected - 1
                    output_image[i, j] = pixels[selected]
                else:
                    output_image[i, j] = np.average(pixels)



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

@njit
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


# LAB 6
@njit(parallel=True)
def adaptive_filter(input_image, n_size, smax):
    output_image = np.zeros(input_image.shape, dtype=np.uint8)
    m, n = input_image.shape

    for i in prange(m):
        for j in prange(n):
            n_size_2 = n_size

            while True:
                step = (int)((n_size_2 - 1) / 2)
                pixels = np.zeros(n_size_2 * n_size_2)

                for i2 in range(n_size_2):
                    for j2 in range(n_size_2):
                        if i - step + i2 >= 0 \
                                and i - step + i2 < input_image.shape[0] \
                                and j - step + j2 >= 0 \
                                and j - step + j2 < input_image.shape[0]:
                            pixels[j2 * n_size_2 + i2] = input_image[i - step + i2, j - step + j2]

                pixels_sorted = np.sort(pixels)
                med = (int)((n_size_2 * n_size_2-1)/2)
                a1 = pixels_sorted[med] - pixels_sorted[0]
                a2 = pixels_sorted[med] - pixels_sorted[n_size_2 * n_size_2-1]
                if(a1>0 and a2<0):
                    b1 = input_image[i, j] - pixels_sorted[0]
                    b2 = input_image[i, j] - pixels_sorted[n_size_2 * n_size_2-1]
                    if(b1>0 and b2<0):
                        output_image[i, j] = pixels[med]
                    else:
                        output_image[i, j] = pixels_sorted[med]
                    break
                else:
                    if(n_size_2 < smax):
                        n_size_2 += 2
                    else:
                        output_image[i, j] = pixels_sorted[med]
                        break

    return output_image

