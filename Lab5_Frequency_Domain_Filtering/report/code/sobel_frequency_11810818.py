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
    # output_spatial += EE326_SUSTech.convolution_3x3(input_image, kernel2)

    io.imsave("Q5_1_spatial_mask.tif", output_spatial)
    output_spatial += input_image
    io.imsave("Q5_1_spatial_filtered.tif", EE326_SUSTech.format_image(output_spatial))

    # Filtering in the Frequency Domain
    input_image = EE326_SUSTech.zero_padding_DFT(input_image)
    input_image = np.fft.fft2(input_image)
    # io.imsave("Q5_1_image_spectrum.tif", EE326_SUSTech.format_image(np.real(input_image)))
    input_image = np.fft.ifftshift(input_image)

    sz = (input_image.shape[0] - kernel1.shape[0], input_image.shape[1] - kernel1.shape[1])
    DFT_kernel_1 = np.pad(kernel1, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    DFT_kernel_2 = np.pad(kernel2, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    DFT_kernel_1_fft = np.fft.fft2(DFT_kernel_1)
    # DFT_kernel_1_fft = np.fft.ifftshift(DFT_kernel_1_fft)
    DFT_kernel_2_fft = np.fft.fft2(DFT_kernel_2)
    # DFT_kernel_2_fft = np.fft.ifftshift(DFT_kernel_2_fft)

    filtered_1 = np.multiply(input_image, DFT_kernel_1_fft)
    filtered_2 = np.multiply(input_image, DFT_kernel_2_fft)



    filtered_1 = np.fft.ifftshift(filtered_1)
    filtered_1 = np.fft.ifft2(filtered_1)
    filtered_1 = np.real(filtered_1)


    filtered_2 = np.fft.ifftshift(filtered_2)
    filtered_2 = np.fft.ifft2(filtered_2)
    filtered_2 = np.real(filtered_2)

    filtered = filtered_1 + filtered_2
    filtered = np.real(filtered)
    filtered -= np.ones(filtered.shape)*np.min(filtered)

    io.imsave("Q5_1_frequency_filtered_1.tif",
              EE326_SUSTech.extract_result_eastsouth((filtered_1)))
    io.imsave("Q5_1_frequency_filtered_2.tif",
              EE326_SUSTech.extract_result_eastsouth((filtered_2)))
    io.imsave("Q5_1_frequency_filtered_3.tif",
              EE326_SUSTech.extract_result_eastsouth((filtered)))

if __name__ == '__main__':
    input_image = io.imread("Q5_1.tif")
    sobel_filter_11810818(input_image)




