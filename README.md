# EE326:Digital Image Processing LAB

## At the begining

This project is the lab exercise of cource **EE326 Digital Image Processing** in **Southern University of Science and Technology**

For discussion, you can contact me by `yuant2018@mail.sustech.edu.cn`

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

## Lab 4:


## Lab 5:

### Task I
**Implement the Sobel filter to the input images Q5_1.tif in both spatial domain and frequency domain.**

Compare the results. Refer to slides 78 to 81 of Lecture 4

**Steps for Filtering in the Frequency Domain**

1. Given an input image $f(x,y)$ of size M x N, obtain the padding parameters P and Q. Typically, P = 2M and Q = 2N.
2. Form a padded image, $f_p(x,y)$ of size P x Q by appending the necessary number of zeros to f (x,y)
3. Multiply $f_p (x,y)$ by $(-1)^{x+y}$ to center its transform
4. Compute the DFT, F(u,v) of the image from step 3
5. Generate a real, symmetric filter function*, H(u,v), of size P x Q with center at coordinates (P/2, Q/2)
6. Form the product G(u,v) = H(u,v)F(u,v) using array multiplication
7. Obtain the processed image
8. Obtain the final processed result, g(x,y), by extracting the M x N region from the top, left quadrant of $g_p (x,y)$



### Task II
Implement the Gaussian low pass and high pass to the input image Q5_2.tif. results for 𝐷 0 = 30 , 60 , and 160, respectively.

### Task III
Implement the Butterworth notch filters to the input images Q5_3.tif. 114 of Lecture 4.

