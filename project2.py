import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skt
from math import floor, exp, sqrt
from PIL import Image

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
	#print("{}, {}\t{}, {}".format(x1,y1,x2,y2))
	x_diff = (x1 - x2)**2
	y_diff = (y1 - y2)**2

	dist = sqrt(x_diff + y_diff)

	return (dist)

def match_features(m1, m2, I1, I2, l):
	r = 0.7
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

		E1 = sum_sq_errors[0][1]
		E2 = sum_sq_errors[1][1]

		#print("{}\t{}".format(E1, E2))
		if (E1 < r * E2):
			matched.append([[p1x, p1y], sum_sq_errors[0][0]])

	return matched

def sum_sq_error(p1, p2):
	sum = 0

	for i, j in zip(p1, p2):
		sum += (i - j)**2

	return (sum)


def compute_homography(sample): 
	cnt = len(list(sample)) 
	u = sample[:, :1]
	v = sample[:, 1:2]
	up = sample[:, 2:3]
	vp = sample[:, 3:4]

	A = np.zeros((2*len(u), 9))

	for i in range(len(u)):
		A[2*i, :]   = [0, 0, 0, -u[i], -v[i], -1, vp[i]*u[i], vp[i]*v[i], vp[i]]
		A[2*i+1, :] = [u[i], v[i], 1, 0, 0, 0, -up[i]*u[i], -up[i]*v[i], -up[i]]

	U,Sigma,Vt = np.linalg.svd(A)

	homog = Vt[-1].reshape(3,3)

	return homog

def RANSAC(number_of_iterations,matches,n,r,d):
	H_best = np.array([[1,0,0],[0,1,0],[0,0,1]])
	list_of_inliers = []
	inlier_best = 0
	
	for i in range(number_of_iterations):
		# 1. Select a random sample of length n from the matches
		print(len(matches))
		np.random.shuffle(matches)
		samples = np.array(matches[:n])
		test = np.array(matches[n:])

		# 2. Compute a homography based on these points using the methods given above
		H_current = compute_homography(samples)

		# 3. Apply this homography to the remaining points that were not randomly selected
		test_I1 = test[:, :2]
		test_I1 = np.column_stack((test_I1, np.ones(len(test_I1))))
		test_I2 = test[:, 2:4]
		
		predicted = (H_current @ test_I1.T).T
		#print(predicted)
		predicted /= predicted[:,2][:,np.newaxis]
		predicted = predicted[:,:2]
		#print(test_I1)
		#print(predicted)
		#print(test_I2)

		# 4. Compute the residual between observed and predicted feature locations
		R = []
		for obs, kwn in zip(predicted, test_I2):
			#print(*obs, *kwn)
			R.append(point_distance(*obs, *kwn))


		current_inliers = []
		for p1, p2 in zip(test_I1, predicted):
			current_inliers.append([p1[:2], p2])

		#print(R)
		in_cnt = 0
		# 5. Flag predictions that lie within a predefined distance r from observations as inliers

		for x in R:
			if x < r:
				in_cnt +=1
		# 6. If number of inliers is greater than the previous best
		#    and greater than a minimum number of inliers d, 
		#    7. update H_best
		#    8. update list_of_inliers
		#print(in_cnt)
		if (in_cnt > inlier_best and in_cnt > d):
			H_best = H_current
			inlier_best = in_cnt
			list_of_inliers = current_inliers
		pass
	
	return H_best, list_of_inliers