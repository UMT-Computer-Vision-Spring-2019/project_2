import numpy as np
import matplotlib.pyplot as plt
from convoluution import *
I = plt.imread('chessboard.png')

Su = np.matrix(
[[-1, 0, 1],
[-2, 0, 2],
[-1,0,1]])

w = np.matrix([
[0.023528,	0.033969,	0.038393,	0.033969,	0.023528],
[0.033969,	0.049045,	0.055432,	0.049045,	0.033969],
[0.038393,	0.055432,	0.062651,	0.055432,	0.038393],
[0.033969,	0.049045,	0.055432,	0.049045,	0.033969],
[0.023528,	0.033969,	0.038393,	0.033969,	0.023528]
])

Iu = convolve_no_edge(I, Su)
Iv = convolve_no_edge(I, Su.T)

Iuu = convolve_no_edge(np.multiply(Iu, Iu), w)
Ivv = convolve_no_edge(np.multiply(Iv, Iv), w)
Iuv = convolve_no_edge(np.multiply(Iu, Iv), w)

H = np.divide(np.multiply(Iuu, Ivv) - np.multiply(Iuv, Iuv), Iuu + Ivv + 1e-10)

plt.imshow(H)
plt.show() 
