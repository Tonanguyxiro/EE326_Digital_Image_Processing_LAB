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
    input_image_padded = EE326_SUSTech.transform_centering(input_image_padded)

    sz = (input_image_padded.shape[0] - kernel1.shape[0], input_image_padded.shape[1] - kernel1.shape[1])
    DFT_kernel_1 = np.pad(kernel1, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    DFT_kernel_2 = np.pad(kernel2, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    input_image_padded_fft = np.fft.fft2(input_image_padded)
    DFT_kernel_1_fft = np.fft.fft2(DFT_kernel_1)
    DFT_kernel_2_fft = np.fft.fft2(DFT_kernel_2)



    m1, n1 = input_image_padded.shape
    filtered_1 = np.real(np.fft.ifft2(input_image_padded_fft * DFT_kernel_1_fft))
    # [m1-600:m1, n1-600:n1]
    filtered_1 = EE326_SUSTech.extract_result(filtered_1)
    filtered_1 = EE326_SUSTech.transform_centering(filtered_1)
    filtered_2 = np.real(np.fft.ifft2(np.fft.fft2(input_image_padded) * DFT_kernel_2_fft))[m1-600:m1, n1-600:n1]
    filtered_2 = EE326_SUSTech.transform_centering(filtered_2)

    io.imsave("Q5_1_frequency_mask.tif", EE326_SUSTech.format_image(filtered_1 + filtered_2))
    io.imsave("Q5_1_frequency_filtered.tif", EE326_SUSTech.format_image(filtered_1 + filtered_2 + input_image))
    io.imsave("Q5_1_image_spectrum.tif", EE326_SUSTech.format_image(np.real(input_image_padded_fft)))
    io.imsave("Q5_1_kernel_spectrum.tif", EE326_SUSTech.format_image(np.real(DFT_kernel_1_fft + DFT_kernel_2_fft)))
    io.imsave("Q5_1_image_filtered_spectrum.tif", EE326_SUSTech.format_image(
        np.real(input_image_padded_fft * (DFT_kernel_1_fft+DFT_kernel_2_fft))))


if __name__ == '__main__':
    input_image = io.imread("Q5_1.tif")
    sobel_filter_11810818(input_image)




