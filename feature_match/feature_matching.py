import numpy as np
from matplotlib import pyplot as plt
from feature_match.keypoint_detection import harris_corner_detection, pull_local_maxima

I = plt.imread('statue_of_liberty.jpg')
I = I.mean(axis=2)






