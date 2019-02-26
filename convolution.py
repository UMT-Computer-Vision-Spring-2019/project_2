import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# convolve image g with kernel h
def convolve(g, h, std=False, color_channel=False):
    g = g.astype(dtype='float32')

    # image dimensions and color channels (add color channel if grayscale)
    g_shape = None
    if (len(g.shape) == 2):
        g_shape = g.shape
        height, width = g.shape
        g = g.reshape((height, width, 1))
    height, width, colors = g.shape
    g_h = np.zeros_like(g)

    # size of neighborhood
    size = (int)((h.shape[0]-1)/2)

    # iterate each color band
    for color in range(colors):
        # iterate each (u,v) pixel (offset 1 from edges)
        for u in range(size, height-size):
            for v in range(size, width-size):
                g_h[u,v,color] = convolve_pixel(g, h, size, u, v, color)

    if std == True:
        standardize_color_channel(g, width, height, colors, size)

    # reshape to grayscale if necessary
    if (g_shape != None):
        g_h = g_h.reshape(g_shape)

    return g_h

# find the convolution of (g*h)(u,v) at pixel [u,v,color]
def convolve_pixel(g, h, size, u, v, color):
    g_neighborhood = g[u-size:u+size+1, v-size:v+size+1, color]
    g_h = 0

    # convolve over 3x3 neighborhood
    g_h = np.multiply(g_neighborhood,h)
    g_h = g_h.sum()

    return g_h

# standardize color channels over range (0,255)
def standardize_color_channel(g, width, height, colors, size):
    # highest and lowest intensity
    I_max = None
    I_min = None
    for color in range(colors):
        for u in range(size, height-size):
            for v in range(size, width-size):
                if I_min == None or g[u,v,color] < I_min:
                    I_min = g[u,v,color]
                if  I_max == None or g[u,v,color] > I_max:
                    I_max = g[u,v,color]
    I_range = I_max - I_min
    for color in range(colors):
        for u in range(1, height-1):
            for v in range(1, width-1):
                g[u,v,color] = (255 / I_range) * (g[u,v,color] + I_min)
    return g

# Linear (Box) Blur kernel
def linear_kernel(size):
    n = 1+2*size
    h = np.zeros((n, n))

    for j in range(-size, size+1):
        for k in range(-size, size+1):
            h[j+size,k+size] = 1

    # Calculate Unity constant
    Z = 1/np.sum(h)
    h = Z * h

    return h

# Gaussian Blur kernel
def gaussian_kernel(sigma, size):
    n = 1+2*size
    h = np.zeros((n, n))

    for j in range(-size, size+1):
        for k in range(-size, size+1):
            h[j+size,k+size] = np.exp(-1 * (np.power(j,2) + np.power(k,2)) / (2 * np.power(sigma,2)))

    # Calculate Unity constant
    Z = 1/np.sum(h)
    h = Z * h

    return h

# Sobel Operator kernel (u), for Edge Detection
def sobel_u_kernel():
    h = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    return h

# Sobel Operator kernel (v), for Edge Detection
def sobel_v_kernel():
    h = np.array([[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]])
    return h

# convert image to grayscale using "perceptual luminance-preserving formula"
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#############################################################################
##############################       MAIN     ###############################
#############################################################################

# # import image
# img_color = plt.imread('noisy_big_chief.jpeg')
# img_gray = img_color.mean(axis=2)
#
# # show both image
# # fig, ax = plt.subplots(1, figsize=(12,8))
# # plt.imshow(img_color)
# # plt.show()
# # plt.imshow(img_gray, cmap='gray')
# # plt.show()
# # plt.imsave('bnc_gray.jpeg', img_gray, cmap='gray')
#
# # Linear Blur/Smoothing
# h_linear = linear_kernel(5)
# img_linear = convolve(img_color, h_linear).astype(dtype='uint8')
# plt.imshow(img_linear)
# plt.show()
# plt.imsave('bnc_linear.jpeg', img_linear)
#
# # Gaussian Blur/Smoothing
# h_gaussian = gaussian_kernel(3, 5)
# img_gaussian = convolve(img_color, h_gaussian).astype(dtype='uint8')
# plt.imshow(img_gaussian)
# plt.show()
# plt.imsave('bnc_gaussian.jpeg', img_gaussian)
#
# # Sobel-u Edge Detection
# h_sobelu = sobel_u_kernel()
# img_sobelu = convolve(img_gray, h_sobelu)
# plt.imshow(img_sobelu, cmap='gray')
# plt.show()
# plt.imsave('bnc_sobelu.jpeg', img_sobelu, cmap='gray')
#
# # Sobel-v Edge Detection
# h_sobelv = sobel_v_kernel()
# img_sobelv = convolve(img_gray, h_sobelv)
# plt.imshow(img_sobelv, cmap='gray')
# plt.show()
# plt.imsave('bnc_sobelv.jpeg', img_sobelv, cmap='gray')
