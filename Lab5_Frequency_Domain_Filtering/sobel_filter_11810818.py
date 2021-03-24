import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import EE326_SUSTech


def sobel_filter_11810818(input_image):
    m, n = input_image.shape
    kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Filtering in the spatial Domain
    output_spatial = EE326_SUSTech.convolution_3x3(input_image, kernel1)
    output_spatial += EE326_SUSTech.convolution_3x3(input_image, kernel2)

    io.imsave("Q5_1_spatial_mask.tif", EE326_SUSTech.format_image(output_spatial))

    output_spatial += input_image
    io.imsave("Q5_1_spatial_filtered.tif", EE326_SUSTech.format_image(output_spatial))

    # Filtering in the Frequency Domain
    input_image_padded = EE326_SUSTech.zero_padding_DFT(input_image)

    sz = (input_image_padded.shape[0] - kernel1.shape[0], input_image_padded.shape[1] - kernel1.shape[1])
    DFT_kernel_1 = np.pad(kernel1, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    DFT_kernel_2 = np.pad(kernel2, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')

    m1, n1 = input_image_padded.shape
    filtered_1 = np.real(np.fft.ifft2(np.fft.fft2(input_image_padded) * np.fft.fft2(DFT_kernel_1)))[m1-600:m1, n1-600:n1]
    filtered_2 = np.real(np.fft.ifft2(np.fft.fft2(input_image_padded) * np.fft.fft2(DFT_kernel_2)))[m1-600:m1, n1-600:n1]

    io.imsave("Q5_1_frequency_mask.tif", EE326_SUSTech.format_image(filtered_1 + filtered_2))
    io.imsave("Q5_1_frequency_filtered.tif", EE326_SUSTech.format_image(filtered_1 + filtered_2 + input_image))


if __name__ == '__main__':
    input_image = io.imread("Q5_1.tif")
    sobel_filter_11810818(input_image)




