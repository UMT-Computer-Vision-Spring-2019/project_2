import numpy as np
import matplotlib.pyplot as plt
from math import floor, exp, sqrt
from PIL import Image

I = plt.imread('chessboard.png')

#I = I.mean(axis=2) # Use this if the image is RGB

def convolve(g, h, f=0):
	(U, V) = g.shape
	(j, k) = h.shape

	x = floor(j/2)
	y = floor(k/2)

	start_x, start_y, end_x, end_y = (0, 0, 0, 0)

	C = np.zeros((U - (2 * x), V - (2 * y)))
	start_x = x
	end_x = (U - x)
	start_y = y
	end_y = (V - y)

	for u in range(start_x, end_x):
		for v in range(start_y, end_y):
			patch = g[u-x:u+x+1:1,v-y:v+y+1:1]
			C[u - 1][v - 1] = apply_kernel(patch, h)

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
			if d[x][y] == 0:
				d[x][y] = 1
	
	H = np.divide(c, d)

	return (H)


def local_maxima(H, j, k):
	X, Y = H.shape
	maxima = []

	for x in range(0, X, j):
		for y in range(0, Y, k):
			#print("({}, {})\t({}, {})".format(x, x+3, y, y+3))
			patch = H[x:x+3:1, y:y+3:1]
			#print(patch)
			max_x, max_y = np.unravel_index(patch.argmax(), patch.shape)
			max_x += x
			max_y += y

			#print("({}, {})".format(max_x, max_y))
			maxima.append([max_x, max_y, H[max_x][max_y]])

	maxima.sort(key=lambda x: x[2]) # sort by harris response intensity
	maxima.reverse()
	amt = int(len(maxima) * 0.15)
	maxima = maxima[:amt] # take the top n%
	
	suppressed = adaptive_suppression(maxima)
	suppressed.sort(key=lambda x: x[2]) # sort by distance
	suppressed.reverse()
	
	# take the top n from the adaptive suppression
	x = [suppressed[i][0] for i in range(70)]
	y = [suppressed[i][1] for i in range(70)]
	
	return ([x, y])


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

		suppressed.append([closest_x, closest_y, dist])

	return (suppressed)

def point_distance(x1, y1, x2, y2):
	x_diff = (x1 - x2)**2
	y_diff = (y1 - y2)**2

	dist = sqrt(x_diff + y_diff)

	return (dist)

H = harris_response(I)

maxima = local_maxima(H, 3, 3)


plt.imshow(I, cmap=plt.cm.gray)
plt.scatter(*maxima)
plt.show()