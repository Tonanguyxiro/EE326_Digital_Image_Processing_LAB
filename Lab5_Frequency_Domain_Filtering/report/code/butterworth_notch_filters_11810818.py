import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mplimg
import EE326_SUSTech


def butterworth_notch_filters_11810818(input_image):
    input_image = EE326_SUSTech.zero_padding_DFT(input_image)
    x, y = input_image.shape

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)

    show_image = np.abs(input_image)
    mplimg.imsave("Q5_2_spectrum.png",
                  show_image,
                  cmap=cm.gray)

    for sigma in [10, 30, 60, 90, 120, 160]:
        for n in [1, 2, 3]:
            centers = [
                [109, 87],
                [109, 170],
                [115, 330],
                [115, 412],
                [227, 405],
                [227, 325],
                [223, 162],
                [223, 79]
            ]

            filter_lowpass = np.zeros([x, y])

            for center in centers:
                filter_lowpass += EE326_SUSTech.butterworth_filter(x, y, center, n, sigma)

            filter_highpass = np.ones([x, y]) - np.clip(filter_lowpass, 0.00001, 0.99999)
            output_image = np.multiply(input_image, filter_highpass)

            mplimg.imsave("Q5_3_spectrum_filtered_" + str(n) + "_" + str(sigma) + ".png",
                          (np.abs(output_image)),
                          cmap=cm.gray)

            output_image = np.fft.fftshift(output_image)
            output_image = np.abs(np.fft.ifft2(output_image))
            output_image = EE326_SUSTech.extract_result_westnorth(output_image)

            mplimg.imsave("Q5_3_filter_" + str(n) + "_" + str(sigma) + ".png",
                          (np.abs(filter_highpass)),
                          cmap=cm.gray)

            mplimg.imsave("Q5_3_" + str(n) + "_" + str(sigma) + ".png",
                          (np.abs(output_image)),
                          cmap=cm.gray)


if __name__ == '__main__':
    butterworth_notch_filters_11810818(io.imread("Q5_3.tif"))