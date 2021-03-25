import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import EE326_SUSTech


def butterworth_notch_filters_11810818(input_image):

    input_image = EE326_SUSTech.zero_padding_DFT(input_image)
    x, y = input_image.shape
    input_image = np.fft.fft2(input_image)

    for sigma in [1, 10, 30]:
        n = 4
        filter_lowpass = EE326_SUSTech.butterworth_filter(x, y, n, sigma)
        dft_filter_lowpass = np.fft.fft2(filter_lowpass)
        output_image_frquency = input_image * dft_filter_lowpass
        output_image = np.real(np.fft.ifft2(output_image_frquency))

        io.imsave("Q5_3_lowpass_filter_" + str(sigma) + ".tif", EE326_SUSTech.format_image(filter_lowpass))
        io.imsave("Q5_3_lowpass_" + str(sigma) + ".tif",
                  EE326_SUSTech.extract_result(EE326_SUSTech.format_image(output_image)))

        filter_highpass = np.ones((x, y)) * np.max(filter_lowpass) - filter_lowpass
        dft_filter_highpass = np.fft.fft2(filter_highpass)
        output_image_frquency = input_image * dft_filter_highpass
        output_image = np.real(np.fft.ifft2(output_image_frquency))

        io.imsave("Q5_3_highpass_" + str(sigma) + ".tif",
                  EE326_SUSTech.extract_result(EE326_SUSTech.format_image(output_image)))
        io.imsave("Q5_3_hignpass_filter_" + str(sigma) + ".tif", EE326_SUSTech.format_image(filter_highpass))


if __name__ == '__main__':
    butterworth_notch_filters_11810818(io.imread("Q5_3.tif"))