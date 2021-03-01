# EE326:Digital Image Processing LAB

## At the begining

This project is the lab exercise of cource **EE326 Digital Image Processing** in **Southern University of Science and Technology**

## LAB 1: Numpy and matplotlib

In this course we will learn to install anaconda and use numpy as well as matplotlib which you cannot use in the following labs.

## LAB 2: Image Interpolation

In this course we will try to use different algorithms of **Image Interpolation** to resize the image.

1. Use nearest neighbor interpolation and bilinear interpolation to interpolate a grey scale image.
2. Use Python function “interp2” from packet “scipy” or your own written algorithm to interpolate a grey scale image by using bicubic interpolation.

## LAB 3: Spatial Transforms

In the lab, `input_image` is the file name of the input image, `output_image` is the file name of the output image, `input_hist` and `output_hist` are lists containing the histogram of the input image and output image, and `spec_hist` is a list containing a specified histogram of the input image; `m_size` is the scale of the neighborhood size, and `n_size` is the scale of the filter size.

### Task I:

Implement the histogram equalization to the input images Q3_1_1.tif and Q3_1_2.tif.


### Task II

Specify a histogram for image Q3_2.tif, such that by matching the histogram of Q3_2.tif to the specified one, the image is enhanced. Implement the specified histogram matching to the input image Q3_2.tif. You may refer to the histogram given in the Lecture Notes 3 page 49, but not necessary to use the same one. Illustrate your specified histogram graphically and numerically in your report.

### Task III

Implement the local histogram equalization to the input images Q3_3.tif.


### Task IV

Implement an algorithm to reduce the salt-and-pepper noise of an image. The input image is Q3_4.tif.
