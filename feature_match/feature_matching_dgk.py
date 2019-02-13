import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from feature_match.keypoint_detection import harris_corner_detection

I_1 = plt.imread('outside_1.jpg')
I_2 = plt.imread('outside_2.jpg')

I_1 = I_1.mean(axis=2)
I_2 = I_2.mean(axis=2)

I_1 = resize(I_1, (I_1.shape[0] / 12, I_1.shape[1] / 12), anti_aliasing=True, order=3)
I_2 = resize(I_2, (I_2.shape[0] / 12, I_2.shape[1] / 12), anti_aliasing=True, order=3)

h_1 = harris_corner_detection(I_1)
h_2 = harris_corner_detection(I_2)

plt.imshow(I_1)
plt.scatter(h_1[:, 1], h_1[:, 0], c=h_1[:, 2])
plt.show()

plt.imshow(I_2)
plt.scatter(h_2[:, 1], h_2[:, 0], c=h_2[:, 2])
plt.show()




