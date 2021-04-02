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
    input_image = np.fft.fftshift(input_image)

    for n in [1, 2, 3, 4]:
            sigma = 20
            filter_lowpass = EE326_SUSTech.butterworth_filter(x, y, n, sigma)
            output_image_frquency = np.multiply(input_image, filter_lowpass)
            output_image = np.fft.ifftshift(output_image_frquency)
            output_image = np.real(np.fft.ifft2(output_image))

            # io.imsave("Q5_3_lowpass_filter_n" + str(sigma) + ".tif",
            #           EE326_SUSTech.format_image(filter_lowpass))
            io.imsave("Q5_3_lowpass_n" + str(n) + ".tif",
                      EE326_SUSTech.extract_result_westnorth(EE326_SUSTech.format_image(output_image)))

            # filter_highpass = np.ones((x, y)) - filter_lowpass
            # output_image_frquency = np.multiply(input_image, filter_highpass)
            # output_image = np.fft.ifftshift(output_image_frquency)
            # output_image = np.real(np.fft.ifft2(output_image))
            #
            # # io.imsave("Q5_3_highpass_filter_" + str(sigma) + ".tif",
            # #           EE326_SUSTech.format_image(filter_highpass))
            # io.imsave("Q5_3_highpass_n" + str(n) + ".tif",
            #          EE326_SUSTech.extract_result_westnorth(EE326_SUSTech.format_image(output_image)))
            #


if __name__ == '__main__':
    butterworth_notch_filters_11810818(io.imread("Q5_3.tif"))