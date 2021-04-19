import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mplimg
import EE326_SUSTech as ee


def undergo(a, b, u, v, T):
    u, v = np.meshgrid(np.linspace(1, u, u), np.linspace(1, v, v))
    d = a * u + b * v
    # h = (T / (np.pi * d)) * np.sin(np.pi * d) * np.exp(-1j*(np.pi * d))
    h = (T / (np.pi * d)) * np.sin(d * np.pi) * np.exp(-1 * 1j * np.pi * d)

    return h


def radially_limited_inverse_filtering_11810818(input_image, sigma):
    input_name = input_image
    input_image = io.imread(input_image + ".tiff")
    m, n = input_image.shape

    filter = undergo(0.1, 0.1, m, n, 1)
    inverse_filter = np.reciprocal(filter)
    g = ee.butterworth_filter(m, n, [m/2, n/2], 10, sigma)

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)

    input_image = input_image * inverse_filter * g
    # input_image = input_image * inverse_filter

    input_image = np.fft.ifftshift(input_image)
    output_image = np.real(np.fft.ifft2(input_image))

    return output_image


def wiener_filter_11810818(input_image, sigma, k):
    input_image = io.imread(input_image + ".tiff")
    m, n = input_image.shape
    # input_image = np.pad(input_image, ((0, m), (0, n)))
    # m, n = input_image.shape

    g = ee.butterworth_filter(m, n, [m/2, n/2], 10, sigma)
    filter = undergo(0.1, 0.1, m, n, 1)

    # f = ((1/filter)*(filter*np.conj(filter)/(filter*np.conj(filter) + k))) * g
    # f = ((filter * np.conj(filter) / (filter * np.conj(filter) + k)))

    buf = filter * np.conj(filter)
    f = buf / (filter * (buf + k)) * g

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)

    output_image = input_image * f

    output_image = np.fft.ifftshift(output_image)
    output_image = np.fft.ifft2(output_image)
    output_image = np.real(output_image)

    return output_image


def test_image():
    test = np.ones([100, 100])*255
    test[49:51, 49:51] = [[0, 0], [0, 0]]

    mplimg.imsave("test_input.png",
                  test,
                  cmap=cm.gray)

    filter = undergo(0.1, 0.1, 100, 100, 1)
    test = np.fft.fftshift(np.fft.fft2(test))
    test = test * filter
    test = np.fft.ifft2(np.fft.fftshift(test))

    mplimg.imsave("test_input_burl.png",
                  np.real(test),
                  cmap=cm.gray)

    test = np.fft.fftshift(np.fft.fft2(test))
    test = test * np.reciprocal(filter)
    test = np.fft.ifft2(np.fft.fftshift(test))

    mplimg.imsave("test_input_restore.png",
                  np.real(test),
                  cmap=cm.gray)

if __name__ == '__main__':
    # test_image()

    input_image = "Q6_3_1"

    for sigma in [1, 10, 50, 100]:
        output_name = "plots/" + str(input_image) + "_radially_limited_" + str(sigma) + ".png"
        output_image = radially_limited_inverse_filtering_11810818(input_image, sigma)
        mplimg.imsave(output_name,
                      output_image,
                      cmap=cm.gray)
    print("Finish processing radially limited filtering")

    for sigma in [40, 70, 100]:
        for K in [0.00000001, 0.0001, 0.1]:
            output_name = "plots/" + str(input_image) + "_wiener_" + str(sigma) + "_" + str(K) + ".png"
            output_image = wiener_filter_11810818(input_image, sigma, K)
            mplimg.imsave(output_name,
                          output_image,
                          cmap=cm.gray)
    print("Finish processing wiener filtering")
