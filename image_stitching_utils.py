# Functions to perform image stitching

import numpy as np
import matplotlib.pyplot as plt

####
#Convolution Integral
####
def convolve(im,h):
    """g = original image 
       h = kernal"""
    
    #Kernal Info
    hx,hy = np.shape(h) #shape of the kernal
    half_h = hx//2      #half of the kernal
    
    #Add a fictitous border around image
    g_pad = np.pad(im, pad_width=half_h, mode='edge') #mode='constant', constant_values=0)
    
    #Picture Info
    gx,gy = np.shape(g_pad) #shape of the image
    
    # Something to store convolved product
    conv = np.zeros((gx-half_h*2,gy-half_h*2))

    # Perform the Convolution
    for i in range(half_h, gx-half_h):
        for j in range(half_h, gy-half_h):
            # part of picture that matches the kernal
            pic = g_pad[i-half_h:i+half_h+1, j-half_h:j+half_h+1]
            conv[i-half_h,j-half_h] = np.sum(np.multiply(h,pic))
    return conv

####
#Guassian Kernal
####
def gauss_kernal(j,k,sigma):
    '''j and k are the size
       sigma is standard devation'''
    h = np.ones((j,k))
    half_h = h.shape[0]//2
    for i in range(j):
        for z in range(k):
            h[i,z] = np.exp(-1*((i-half_h)**2+(z-half_h)**2)/(2*sigma**2))        
    h = h/np.sum(h)
    return h

####
#Harris Response
####
def harris_response(im, sobel, gauss):
    """Takes an image, sobel kernal, and gauss kernal"""
    #Convolve with sobel kernal
    Iu = convolve(im,sobel)
    Iv = convolve(im,sobel.T)
    
    # Gradients
    Iuu = convolve(np.multiply(Iu,Iu), gauss) + 1.e-10 #Add small number to avoid dividing by zero
    Ivv = convolve(np.multiply(Iv,Iv), gauss) + 1.e-10 #Also avoid zero division by making larger gaussian kernal size
    Iuv = convolve(np.multiply(Iu,Iv), gauss) + 1.e-10
    
    # Calculate Harris Repsonse
    num = np.multiply(Iuu, Ivv) - np.multiply(Iuv, Iuv)
    den = Iuu + Ivv
    return np.divide(num,den)

####
#Extract Local Maxima
####
def extract_pts(H):
    '''Requires the Harris Response of an Image'''
    m,n = H.shape
    
    # Arrays to store points
    maxpts = []
    
    # Loop through and compare points
    for i in np.arange(m-1):
        for j in np.arange(n-1):
            pt = H[i,j].astype(float)     # this indexing might need to change
            
            if pt > H[i-1,j-1] and pt > H[i-1,j] and pt > H[i-1,j+1] and\
            pt > H[i,j-1] and pt > H[i,j+1] and\
            pt > H[i+1,j-1] and pt > H[i+1,j] and pt > H[i+1,j+1]: 
        
                maxpts.append([j,i,pt])
            else:
                continue
    maxpts.sort(key=lambda x: x[2])  #Sort based on harris response - smallest to largest
    maxpts.reverse()
    return np.array(maxpts)


####
# Non-Maximal Suprresion
####
def nonmax_suppression(maxpts, n):
    '''Takes the outputs of extract_points
       n is the number of points to keep'''
    trim = maxpts[:n,:]
    return trim

####
# Adaptive Supression
####

def point_distance(x1, y1, x2, y2):
        x_diff = (x1 - x2)**2
        y_diff = (y1 - y2)**2

        dist = np.sqrt(x_diff + y_diff)

        return (dist)


def adaptive_supression(maxpts, n):
    '''Takes list of max points from either 
	    extract_points or nonmax_suppression
        n is the number of points to keep'''
    #Coordinates of corners
    x, y = maxpts[:,0], maxpts[:,1]
    
    #Place to store the points
    suppressed=[]
    
    for i in range(len(maxpts)):
        # for every point
        xi, yi = maxpts[i,0], maxpts[i,1]
        Hi = maxpts[i,2]
        closest_dist = np.inf
        dist = 0

        for j in range(len(maxpts)):
            xj, yj = maxpts[j,0], maxpts[j,1]
            Hj = maxpts[j,2]
            if 0.9*Hj > Hi:
                dist = point_distance(xi, yi, xj, yj)
                if dist < closest_dist:
                    closest_dist = dist            
        suppressed.append(closest_dist)
    
    # Sort the points based on distance
    xf = x[np.argsort(suppressed)]
    xf = xf[(len(x)-n):]
    yf = y[np.argsort(suppressed)]
    yf = yf[(len(y)-n):]
    
    return np.column_stack((xf, yf))

####
# Feature Matching
####

def descriptor(im, corners, bnd):
    '''im = original image
       pxls are detected points
       bnd is the size of the descriptor square'''
    #Shape of the imnage
    m,n = np.shape(im)
    
    #Bounds
    bb = bnd//2
    
    #Hold Descriptors
    des=[]
    pix_vals = []
    
    for i, j in zip(corners[:,0], corners[:,1]):
        i, j = int(i), int(j)
        #pixel = im[i-bb : i+bb+1, j-bb : j+bb+1]  # WHY!
        pixel = im[j-bb : j+bb+1, i-bb : i+bb+1]  # WHY!
       
        if pixel.shape[0]==bnd and pixel.shape[1]==bnd:
            des.append(pixel)
            pix_vals.append((i,j))
            
    return np.asarray(des), np.asarray(pix_vals)

def dhat(patch):
    return (patch - patch.mean()) / np.std(patch)

def error(dhat1, dhat2):
    err = (dhat1 - dhat2)**2
    return np.sum(err)

# Run this function with output from adaptive_suppression to et
# the patching points between to images
def image_match_ths(im1, im2, corners1, corners2, bnd, r):
    '''Find matching points between 2 images
       des are the descriptors of each image'''
    
    dcs1, px1 = descriptor(im1, corners1, bnd=21)
    dcs2, px2 = descriptor(im2, corners2, bnd=21)
    
    match = []
    c1 = []
    c2 = []
    for i in range(len(dcs1)):
        err1 = np.inf
        err2 = np.inf
        d1 = dcs1[i].flatten()
        dhat1 = dhat(d1)
        
        for j in range(len(dcs2)):
            d2 = dcs2[j].flatten()
            dhat2 = dhat(d2)
            ei = error(dhat1, dhat2)
            
            if ei < err2:
                err2=ei    
                
                if err2<err1:
                    err2 = err1  #Bumps the previous lowest to 2nd place
                    err1 = ei    #Assigns new lowest, the new one to beat 
                    match_num = [i,j]
                    im_1 = px1[i]
                    im_2 = px2[j]
         
        if err1<r*err2:
            c1.append(im_1) #Corner Coordinates in Image 1
            c2.append(im_2) #Matching Corners in image 2
            match.append(match_num)
        
    return np.asarray(match), np.asarray(c1), np.asarray(c2)

def match_plot(im1, im2, pts_im1, pts_im2, corners1, corners2, w, h):

    m,n = im1.shape

    #Clip two fingures together
    clip_together = np.column_stack((im1, im2))

    #Plots
    fig, ax = plt.subplots(1,1, figsize=(w,h))
    ax.imshow(clip_together, cmap='gray')
    
    #Plot corresponding points
    for i in range(len(pts_im1)):
        ax.plot([pts_im1[:,0], pts_im2[:,0]+n], [pts_im1[:,1], pts_im2[:,1]])
    
    # Plot the original corners
    ax.plot(corners1[:,0], corners1[:,1], 'ro')
    ax.plot(corners2[:,0]+n, corners2[:,1], 'bo')
    plt.show()
    
    return


