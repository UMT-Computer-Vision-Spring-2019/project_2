import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# local imports
from convolution import *

# find local maxima of Harris Response matrix
def local_maxima(g, h, size):
    return 0

# Check if a given pixel is a local maxima
def check_if_local_maxima(H, size, width, height, u, v):
    # test pixel Harris score
    px = H[u,v]
    # iterate over neighborhood
    for j in range(u-size, u+size+1):
        for k in range(v-size, v+size+1):
            if j<0 or k<0 or j>height-1 or k>width-1:
                continue
            # skip itself
            if j==u and k==v:
                continue
            # if pixel of greater score found, not maxima
            elif px < H[j,k]:
                return False
    # else, is maxima
    return True

def sum_sq_error(p1, p2):
	sum = 0

	for i, j in zip(p1, p2):
		sum += (i - j)**2

	return (sum)
