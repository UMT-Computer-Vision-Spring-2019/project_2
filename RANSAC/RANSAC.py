import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skt
from homography import *

def apply_homography(X, H):
  Xprime = (H.dot(X.T)).T
  Xprime/=Xprime[:,2][:,np.newaxis]
  return Xprime

def RANSAC(number_of_iterations,matches,n,r,d):
    H_best = np.array([[1,0,0],[0,1,0],[0,0,1]])
    list_of_inliers = np.zeros((2, d, 2))
    # for 3D projection
    matches = np.insert(matches, 2, 1, axis=2)
    for i in range(number_of_iterations):
        # 1. Select a random sample of length n from the matches
      indices = np.random.choice(np.arange(0, matches.shape[1]), size=n, replace=False)
      sample = matches[:,indices,:]
        # 2. Compute a homography based on these points using the methods given above
      H = generate_homography(sample[0], sample[1], n=4) 
        # 3. Apply this homography to the remaining points that were not randomly selected
      pred_location = apply_homography(matches[0], H)
        # 4. Compute the residual between observed and predicted feature locations
      residuals = np.sqrt((matches[1,:,0] - pred_location[:,0])**2 + (matches[1,:,1] - pred_location[:,1])**2)
        # 5. Flag predictions that lie within a predefined distance r from observations as inliers
      inliers = matches[:,residuals<=r]
        # 6. If number of inliers is greater than the previous best
        #    and greater than a minimum number of inliers d, 
        #    7. update H_best
        #    8. update list_of_inliers
      if (inliers.shape[1] > list_of_inliers.shape[1]):
        print(inliers.shape[1])
        H_best = H
        list_of_inliers = inliers
    return H_best, list_of_inliers

I_1 = plt.imread('photo_1.jpg')
I_2 = plt.imread('photo_2.jpg')

I_1 = I_1.mean(axis=2)
I_2 = I_2.mean(axis=2)

corners_1 = harris_corner_detection(I_1)
corners_2 = harris_corner_detection(I_2)

descriptors_1 = extract_descriptors(I_1, corners_1, 21)
descriptors_2 = extract_descriptors(I_2, corners_2, 21)

matching_indices = get_matches(descriptors_1, descriptors_2 , .5).astype(int)

matching_corners_1 = corners_1[matching_indices[:,0]]
matching_corners_2 = corners_2[matching_indices[:,1]]

matches = np.array([matching_corners_1, matching_corners_2])

H_best, inliers = RANSAC(10000, matches, 10, 30, 4)

# Create a projective transform based on the homography matrix $H$
proj_trans = skt.ProjectiveTransform(H_best)

# Warp the image into image 1's coordinate system
I_2_transformed = skt.warp(I_2,proj_trans)

plt.imshow(I_1)
plt.imshow(I_2_transformed, alpha=.5)
plt.show()
