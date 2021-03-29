import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mplimg
import EE326_SUSTech


def sobel_filter_11810818(input_image):
    m, n = input_image.shape
    kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel = kernel1 + kernel2

    # Filtering in the spatial Domain
    # output_spatial = EE326_SUSTech.convolution_3x3(input_image, kernel)
    # plt.imsave("Q5_1_spatial.png", output_spatial, cmap=cm.gray)

    # Filtering in the Frequency Domain
    input_image = np.pad(input_image, ((0, m), (0, n)))
    input_image = np.fft.fft2(input_image)

    mplimg.imsave("Q5_1_spectrum_noshift.png",
                  np.log(np.abs(input_image)),
                  cmap=cm.gray)
    input_image = np.fft.fftshift(input_image)
    mplimg.imsave("Q5_1_spectrum.png",
                  np.log(np.abs(input_image)),
                  cmap=cm.gray)
    sz = (input_image.shape[0] - kernel1.shape[0],
          input_image.shape[1] - kernel1.shape[1])
    kernel = np.pad(kernel,
                    (((sz[0] + 1) // 2, sz[0] // 2),
                     ((sz[1] + 1) // 2, sz[1] // 2)))
    kernel_fft = np.fft.fft2(kernel)
    kernel_fft = np.fft.fftshift(kernel_fft)

    filtered = np.multiply(input_image, kernel_fft)
    mplimg.imsave("Q5_1_spectrum_filtered.png",
                  np.log(np.abs(input_image)),
                  cmap=cm.gray)
    filtered = np.fft.ifft2(np.fft.fftshift(filtered))

    output_image = EE326_SUSTech.extract_result_eastsouth(filtered)
    output_image = np.abs(output_image)

    mplimg.imsave("Q5_1_frequency.png",
                  np.uint8(output_image),
                  cmap=cm.gray)


if __name__ == '__main__':
    sobel_filter_11810818(io.imread("Q5_1.tif"))




