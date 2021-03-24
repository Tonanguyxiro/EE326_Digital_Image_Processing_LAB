import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import EE326_SUSTech


def gaussian_pass_11810818(input_image):
    x, y = input_image.shape

    input_image = np.fft.fft2(input_image)

    for sigma in [30, 60, 160]:
        filter_lowpass = EE326_SUSTech.gaussian_filter((x, y), sigma)
        dft_filter_lowpass = np.fft.fft2(filter_lowpass)
        output_image_frquency = input_image * dft_filter_lowpass
        output_image = np.real(np.fft.ifft2(output_image_frquency))

        io.imsave("Q5_2_lowpass_"+str(sigma)+".tif", EE326_SUSTech.format_image(output_image))

        filter_highpass = np.ones((x, y)) - filter_lowpass
        dft_filter_highpass = np.fft.fft2(filter_highpass)
        output_image_frquency = input_image * dft_filter_highpass
        output_image = np.real(np.fft.ifft2(output_image_frquency))

        io.imsave("Q5_2_highpass_" + str(sigma) + ".tif", EE326_SUSTech.format_image(output_image))


if __name__ == '__main__':
    input_image = io.imread("Q5_2.tif")
    gaussian_pass_11810818(input_image)