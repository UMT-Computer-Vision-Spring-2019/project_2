from Stitcher import *
from convolution import *

im1_fname = 'room1.jpg'
im2_fname = 'room2.jpg'

# Convert both images to gray scale
im_1 = np.mean(plt.imread(im1_fname), -1)
im_2 = np.mean(plt.imread(im2_fname), -1)

# Note: can uncomment this to speed things up
# Perform Gaussian blur to eliminate aliasing
# gauss = Filter.make_gauss((3, 3), 1)
# im_1 = convolve(im_1, gauss)
# im_2 = convolve(im_2, gauss)

# Down sample image to speed things up a bit
# im_1 = block_reduce(im_1, (2, 2), func=np.mean)
# im_2 = block_reduce(im_2, (2, 2), func=np.mean)

st = Stitcher(im_1, im_2)

stitched_image = st.stitch()
plt.imshow(stitched_image, cmap='gray')
plt.show()