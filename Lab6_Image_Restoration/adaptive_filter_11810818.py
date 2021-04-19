import numpy as np
import numpy.fft
from skimage import io, data
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mplimg
from numba import njit,prange
import EE326_SUSTech as ee
import time

@njit(parallel=True)
def adaptive_11810818(input_image, n_size, smax):
    output_image = np.zeros(input_image.shape, dtype=np.uint8)
    m, n = input_image.shape

    for i in prange(m):
        for j in prange(n):
            n_size_2 = n_size

            while True:
                step = (int)((n_size_2 - 1) / 2)
                pixels = np.zeros(n_size_2 * n_size_2)

                for i2 in range(n_size_2):
                    for j2 in range(n_size_2):
                        if i - step + i2 >= 0 \
                                and i - step + i2 < input_image.shape[0] \
                                and j - step + j2 >= 0 \
                                and j - step + j2 < input_image.shape[0]:
                            pixels[j2 * n_size_2 + i2] = input_image[i - step + i2, j - step + j2]

                pixels_sorted = np.sort(pixels)
                med = (int)((n_size_2 * n_size_2-1)/2)
                a1 = pixels_sorted[med] - pixels_sorted[0]
                a2 = pixels_sorted[med] - pixels_sorted[n_size_2 * n_size_2-1]
                if(a1>0 and a2<0):
                    b1 = input_image[i, j] - pixels_sorted[0]
                    b2 = input_image[i, j] - pixels_sorted[n_size_2 * n_size_2-1]
                    if(b1>0 and b2<0):
                        output_image[i, j] = pixels[med]
                    else:
                        output_image[i, j] = pixels_sorted[med]
                    break
                else:
                    if(n_size_2 < smax):
                        n_size_2 += 2
                    else:
                        output_image[i, j] = pixels_sorted[med]
                        break

    return output_image

if __name__ == '__main__':
    for i in [1
        , 2
        , 3
        , 4
              ]:
        start_time = time.time()
        input_image = "Q6_1_" + str(i)
        output_name = "plots/" + str(input_image) + "_adaptive.png"
        input_image = io.imread(input_image + ".tiff")
        output_image = adaptive_11810818(input_image, 3, 20)
        print(time.time() - start_time)
        mplimg.imsave(output_name,
                      output_image,
                      cmap=cm.gray)

        print("Finish processing " + str(i))