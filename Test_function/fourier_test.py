import numpy as np
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt

input_image = io.imread("Q5_1.tif")
input_image = np.fft.fft2(input_image)
input_image = np.fft.fftshift(input_image)

input_image = np.fft.ifftshift(input_image)
input_image = np.fft.ifft2(input_image)

io.imsave("Q5_1_ifft2.tif", np.real(input_image).astype(np.uint8))





