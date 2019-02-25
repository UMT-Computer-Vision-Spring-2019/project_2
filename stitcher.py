import numpy as np
import matplotlib.pyplot as plt

from keypoint_detector import *

class Stitcher(object):
	def __init__(self,image_1,image_2):
		self.images = [image_1,image_2]

	def find_keypoints(self, img):
		"""
		Step 1: This method locates features that are "good" for matching.  To do this we will implement the Harris 
		corner detector
		"""
		# import image and convert to grayscale
		I = img
		I = I.mean(axis=2)

		# detect_keypoints(I)
		g = I

		h_sobelu = sobel_u_kernel()
		h_sobelv = sobel_v_kernel()
		h_gaussian = gaussian_kernel(2, 2)

		g_u = convolve(g, h_sobelu)
		g_u2 = np.multiply(g_u, g_u)
		g_u2 = convolve(g_u2, h_gaussian)

		g_v = convolve(g, h_sobelv)
		g_v2 = np.multiply(g_v, g_v)
		g_v2 = convolve(g_v2, h_gaussian)

		g_uv = np.multiply(g_u, g_v)
		g_uv = convolve(g_uv, h_gaussian)

		# Harris Response matrix
		H = np.zeros_like(g)
		# Determinant(g)
		H_det = np.multiply(g_u2, g_v2) - np.multiply(g_uv, g_uv)
		# Trace(g) (small number to prevent division by zero)
		H_tr = g_u2 + g_v2 + 1e-10
		# H = det(g)/tr(g)
		H = np.divide(H_det, H_tr)
		
		# Find local maxima of Harris matrix scores
		height, width = H.shape
		max = []
		# size of neighborhood
		size = 50

		# iterate over pixels
		for u in range(height):
			for v in range(width):
				# add to list if local maxima
				if check_if_local_maxima(H, size, width, height, u, v):
					max.append([u,v])

		for i in range(len(max)):
			plt.scatter(x=max[i][1], y=max[i][0], c='r')

		plt.imshow(I,cmap=plt.cm.gray)
		plt.show()
		
		return max

	def generate_descriptors(self,img,points, l):
		"""
		Step 2: After identifying relevant keypoints, we need to come up with a quantitative description of the 
		neighborhood of that keypoint, so that we can match it to keypoints in other images.
		"""
		l = int(l/2)
		img = img.mean(axis=2)
		
		descriptors = []
		for i in range(len(points)):
			x1 = points[i][0]-l
			x2 = points[i][0]+l+1
			y1 = points[i][1]-l
			y2 = points[i][1]+l+1
			
			if(x1 < 0):
				x1 = 0
			if(y1 < 0):
				y1 = 0
			if(x2 > img.shape[0]):
				x2 = img.shape[0]
			if(y2 > img.shape[1]):
				y2 = img.shape[1]
			
			descriptors.append(img[x1:x2, y1:y2])
		return descriptors
			
		

	def match_keypoints(self):
		"""
		Step 3: Compare keypoint descriptions between images, identify potential matches, and filter likely
		mismatches
		"""
		matches = []
		m1 = myStitcher.find_keypoints(self.images[0])
		m2 = myStitcher.find_keypoints(self.images[1])
		desc1 = myStitcher.generate_descriptors(self.images[0], m1, 21)
		desc2 = myStitcher.generate_descriptors(self.images[1], m2, 21)
		
		for img1 in desc1:
			Evals = []
			for img2 in desc2:
				Evals.append([[img2[0], img2[1]], sum_sq_error(img1.flatten(), img2.flatten())])
			Evals.sort(key=lambda x: x[1])

			matches.append([[img1[0], img1[1]], Evals[0][0]])

					
		return matches


	def find_homography(self):
		"""
		Step 4: Find a linear transformation (of various complexities) that maps pixels from the second image to 
		pixels in the first image
		"""

	def stitch(self):
		"""
		Step 5: Transform second image into local coordinate system of first image, and (perhaps) perform blending
		to avoid obvious seams between images.
		"""

im1 = plt.imread('class_photo1re.jpg')
im2 = plt.imread('class_photo2re.jpg')
myStitcher = Stitcher(im1, im2)
matches = myStitcher.match_keypoints()
print(len(matches))

I = np.concatenate((im1, im2), axis=1)

plt.imshow(I, cmap=plt.cm.gray)

# for i in range(amt):
	# plt.plot([x1[i], x2[i]], [y1[i], y2[i]])

plt.show(block=True)