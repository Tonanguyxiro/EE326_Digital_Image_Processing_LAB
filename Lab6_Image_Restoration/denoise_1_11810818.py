import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mplimg
import EE326_SUSTech as ee


def denoise_11810818(input_image, n_size, mode):
    output_name = "plots/" + str(input_image) + "_denoised.png"
    input_image = io.imread(input_image+".tiff")

    for i in mode:
        output_image = ee.denoise_filter(input_image, n_size, i)

    mplimg.imsave(output_name,
                  output_image,
                  cmap=cm.gray)


if __name__ == '__main__':
    # denoise_11810818("Q6_1_1", 3, ["medium"])
    # denoise_11810818("Q6_1_2", 5, ["medium"])
    denoise_11810818("Q6_1_3", 9, ["medium"])
    # denoise_11810818("Q6_1_4", 9, ["medium"])
