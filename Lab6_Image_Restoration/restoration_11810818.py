import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mplimg
import EE326_SUSTech as ee


def atmosphere_turbulence(shape, k):
    col, row = shape
    u, v = np.meshgrid(np.linspace(0, col-1, col), np.linspace(0, row-1, row))
    u = u - col / 2
    v = v - row / 2
    d = u * u + v * v
    h = np.exp(-(k * (d ** (5/6))))

    return h


def full_inverse_filtering_11810818(input_image):
    input_image = io.imread(input_image + ".tif")
    m, n = input_image.shape

    filter = atmosphere_turbulence(input_image.shape, 0.0025)
    inverse_filter = 1/filter

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)
    input_image = input_image*inverse_filter
    input_image = np.fft.fftshift(input_image)
    output_image = np.abs(np.fft.ifft2(input_image))

    return output_image

def radially_limited_inverse_filtering_11810818(input_image, sigma):
    input_image = io.imread(input_image + ".tif")
    m, n = input_image.shape

    filter = atmosphere_turbulence(input_image.shape, 0.0025)
    inverse_filter = 1 / filter
    g = ee.butterworth_filter(m, n, [m/2, n/2], 10, sigma)

    input_image = np.fft.fft2(input_image)
    output_image = np.abs(input_image)
    mplimg.imsave("image_test.png",
                  np.abs(input_image),
                  cmap=cm.gray)

    input_image = np.fft.fftshift(input_image)


    input_image = input_image * inverse_filter * g
    input_image = np.fft.fftshift(input_image)
    output_image = np.abs(np.fft.ifft2(input_image))

    return output_image

def wiener_filter_11810818(input_image, sigma, k):
    input_image = io.imread(input_image + ".tif")
    m, n = input_image.shape

    g = ee.gaussian_filter(m, n, 30)
    filter = atmosphere_turbulence(input_image.shape, 0.0025)

    f = ((1/filter)*(filter**2/(filter**2 + k*np.ones([m, n])))) * g

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)
    output_image = input_image*f
    output_image = np.fft.fftshift(output_image)
    output_image = np.fft.ifft2(output_image)
    output_image = np.abs(output_image)


if __name__ == '__main__':

    input_image = "Q6_2"
    output_name = "plots/" + str(input_image) + "_full_inverse.png"
    output_image = full_inverse_filtering_11810818(input_image)
    mplimg.imsave(output_name,
                  output_image,
                  cmap=cm.gray)
    print("Finish processing full inverse filtering")

    # for i in [40, 50, 60, 70, 80, 90, 100]:
    for i in [70]:
        output_name = "plots/" + str(input_image) + "_radially_limited" + str(i) + ".png"
        output_image = radially_limited_inverse_filtering_11810818(input_image, i)
        mplimg.imsave(output_name,
                      output_image,
                      cmap=cm.gray)
    print("Finish processing full inverse filtering")

    print("Finish processing full inverse filtering")