import numpy as np
from keypoint_detection import *

def pad_with(vector, pad_width, iaxis, kwargs):
  pad_value = kwargs.get('padder', 0)
  vector[:pad_width[0]] = pad_value
  vector[-pad_width[1]:] = pad_value 
  return vector

def get_min_descriptor_errors(descirptors_1, descriptors_2):
  descriptor_with_least_error = np.zeros((descriptors_1.shape[0], 2))
  for i in range(0, descriptors_1.shape[0]):
    error = float('inf')
    for j in range(0, descriptors_2.shape[0]):
      if (i != j):  
        test_error = np.sum((descriptors_1[i] - descriptors_2[j])**2)
        if (test_error < error):
          descriptor_with_least_error[i][0] = j
          # is robust
          descriptor_with_least_error[i][1] = test_error < error * .3
          error = test_error
  return descriptor_with_least_error

def extract_descriptors(I, corners, l):
  descriptors = np.zeros((corners.shape[0], l/2*2, l/2*2))
  I = np.pad(I, l/2 + 1, pad_with)
  for i in range(0, corners.shape[0]):
    # uuh might have indices swapped
    x = corners[i][0] + l/2
    y = corners[i][1] + l/2
    descriptor = I[int(x-l/2):int(x+l/2),int(y-l/2):int(y+l/2)]
    descriptors[i] = descriptor
  return descriptors
    
I_1 = plt.imread('photo_1.jpg')
I_2 = plt.imread('photo_2.jpg')

I_1 = I_1.mean(axis=2)
I_2 = I_2.mean(axis=2)

corners_1 = harris_corner_detection(I_1)
corners_2 = harris_corner_detection(I_2)

descriptors_1 = extract_descriptors(I_1, corners_1, 21)
descriptors_2 = extract_descriptors(I_2, corners_2, 21)

img = np.zeros((I_1.shape[0], I_1.shape[1] * 2))
img[:,I_1.shape[1]:] = I_2
img[:,:I_1.shape[1]] = I_1

descriptor_with_least_error = get_min_descriptor_errors(descriptors_1, descriptors_2)

for i in range(0, descriptor_with_least_error.shape[0]):
  x1 = corners_1[i][1] 
  y1 = corners_1[i][0]
  
  x2 = I_2.shape[1] + corners_2[int(descriptor_with_least_error[i][0])][1]
  y2 = corners_2[int(descriptor_with_least_error[i][0])][0]
  if (descriptor_with_least_error[i][1]):
    plt.plot([x1, x2], [y1, y2])

plt.scatter(corners_1[:,1], corners_1[:,0], c='blue')
plt.scatter(I_1.shape[1] + corners_2[:,1], corners_2[:,0], c='white')
plt.imshow(img)
plt.show()

