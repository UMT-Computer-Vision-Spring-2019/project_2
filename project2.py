import numpy as np
import matplotlib.pyplot as plt
from math import floor, exp, sqrt
from PIL import Image

I1 = plt.imread('2.jpg')
I2 = plt.imread('1.jpg')

I1 = I1.mean(axis=2) # Use this if the image is RGB
I2 = I2.mean(axis=2)

I = np.concatenate((I1, I2), axis=1)

def convolve(g, h, f=0):
	(V, U) = g.shape
	(j, k) = h.shape
	
	x = floor(j/2) # half of filter x dim
	y = floor(k/2) # half of filter y dim

	C = np.zeros((V, U)) # this is where the result of our convolution will go

	padded = np.zeros((V + (2 * y), U + (2 * x)))
	padded[y:V+1, x:U+1] = g
	g = padded

	for u in range(x, U):
		for v in range(y, V):
			patch = g[v-y:v+y+1:1, u-x:u+x+1:1]
			C[v - 1][u - 1] = apply_kernel(patch, h)

	return (C)

def apply_kernel(patch, h):
	(j, k) = patch.shape
	z = 0
	for x in range(j):
		for y in range(k):
			z+= (patch[x][y] * h[x][y])

	return(z)

def generate_gaussian(j, k, sigma):
	h = np.zeros((j, k))
	Z = 0
	for x in range(j):
		for y in range(k):
			val = exp(-((x - j/2)**2 + (y - k/2)**2) / (2 * sigma**2))
			h[x][y] = val
			Z += val
	
	for x in range(j):
		for y in range(k):
			h[x][y] = Z * h[x][y]

	return (h)

def harris_response(I):
	w = generate_gaussian(3, 3, 2)

	s_u = np.array([[-1, 0, 1],
					[-2, 0, 2],
					[-1, 0, 1]])
	
	s_v = np.array([[-1, -2, -1],
					[ 0,  0,  0],
					[ 1,  2,  1]])

	I_u = convolve(I, s_u)
	I_v = convolve(I, s_v)
	I_uu = convolve(np.multiply(I_u, I_u), w)
	I_vv = convolve(np.multiply(I_v, I_v), w)
	I_uv = convolve(np.multiply(I_u, I_v), w)
	
	a = np.multiply(I_uu, I_vv)
	b = np.multiply(I_uv, I_uv)
	c = np.subtract(a, b)
	d = np.add(I_uu, I_vv)

	(X, Y) = d.shape

	for x in range(X):
		for y in range(Y):
			d[x][y] += 1e-10
	
	H = np.divide(c, d)

	return (H)

def local_maxima(H, j, k):
	Y, X = H.shape
	maxima = []
	for x in range(0, X, j):
		for y in range(0, Y, k):
			#print("({}, {})\t({}, {})".format(x, x+3, y, y+3))
			patch = H[y:y+3:1, x:x+3:1]
			#print(patch)
			max_y, max_x = np.unravel_index(patch.argmax(), patch.shape)
			max_x += x
			max_y += y

			#print("({}, {})".format(max_x, max_y))
			maxima.append([max_x, max_y, H[max_y][max_x]])

	maxima.sort(key=lambda x: x[2]) # sort by harris response intensity
	maxima.reverse()
	amt = int(len(maxima) * 0.15)
	maxima = maxima[:amt] # take the top n%
	
	suppressed = adaptive_suppression(maxima)
	suppressed.sort(key=lambda x: x[2]) # sort by distance
	suppressed.reverse()


	suppressed = suppressed[0:100]
	return (suppressed)

def adaptive_suppression(maxima):
	suppressed = []
	cnt = len(maxima)

	for i in range(cnt):
		# for every point
		(x, y) = (maxima[i][0], maxima[i][1])
		closest_x = x
		closest_y = y
		closest_dist = float('inf')
		dist = 0
		
		higher_response = maxima[0:i] # we know that only 0:i have a higher harris response
		for point in higher_response:
			# for each point that has a higher harris response
			(px, py) = (point[0], point[1])
			dist = point_distance(x, y, px, py)
			
			if dist < closest_dist:
				closest_dist = dist
				closest_x = px
				closest_y = py

		suppressed.append([x, y, closest_dist])

	return (suppressed)

def point_distance(x1, y1, x2, y2):
	x_diff = (x1 - x2)**2
	y_diff = (y1 - y2)**2

	dist = sqrt(x_diff + y_diff)

	return (dist)

def match_features(m1, m2, I1, I2, l):
	matched = []
	V, U = I1.shape

	l = int(l/2)
	
	for p1 in m1:
		(p1x, p1y) = (p1[0], p1[1]) # coordinates of keypoint in image 1

		# descriptor 1 pixel boundaries
		d1_x1 = p1x - l
		d1_x2 = p1x + l + 1
		d1_y1 = p1y - l
		d1_y2 = p1y + l + 1
		d1 = I1[d1_y1:d1_y2:1, d1_x1:d1_x2:1]

		if (d1_x1 < 0 or d1_x2 >= U or d1_y1 < 0 or d1_y2 >= V):
			# if we are too close to the edge of image 1
			continue

		sum_sq_errors = [] # list of potential matched keypoints in the form [[x, y], (sum_sq_error)]

		for p2 in m2:
			(p2x, p2y) = (p2[0], p2[1]) # coordinates of keypoint in image 2

			# descriptor 2 pixel boundaries
			d2_x1 = p2x - l
			d2_x2 = p2x + l + 1
			d2_y1 = p2y - l
			d2_y2 = p2y + l + 1
			d2 = I2[d2_y1:d2_y2:1, d2_x1:d2_x2:1]

			if (d2_x1 < 0 or d2_x2 >= U or d2_y1 < 0 or d2_y2 >= V):
				# if we are too close to the edge of the image 2
				continue

			sum_sq_errors.append([[p2x, p2y], sum_sq_error(d1.flatten(), d2.flatten())])

		sum_sq_errors.sort(key=lambda x: x[1])

		matched.append([[p1x, p1y], sum_sq_errors[0][0]])

	return matched

def sum_sq_error(p1, p2):
	sum = 0

	for i, j in zip(p1, p2):
		sum += (i - j)**2

	return (sum)


H1 = harris_response(I1)
H2 = harris_response(I2)

m1 = local_maxima(H1, 3, 3)
m2 = local_maxima(H2, 3, 3)

blah, x_diff = I1.shape

"""
fx = []
fy = []

for p1, p2 in zip(m1, m2):
	fx.append(p1[0])
	fx.append(p2[0] + x_diff)
	fy.append(p1[1])
	fy.append(p2[1])
"""

matched = match_features(m1, m2, I1, I2, 21)
amt = len(matched)

x1 = [matched[i][0][0] for i in range(amt)]
y1 = [matched[i][0][1] for i in range(amt)]

x2 = [matched[i][1][0] + x_diff for i in range(amt)]
y2 = [matched[i][1][1] for i in range(amt)]

amt = len(matched)

plt.imshow(I, cmap=plt.cm.gray)

#plt.scatter(fx, fy)

for i in range(amt):
	plt.plot([x1[i], x2[i]], [y1[i], y2[i]])

plt.show(block=True)