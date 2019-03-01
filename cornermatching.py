import numpy as np
import math as mt
import random

def convolve(g,h): # h is kernel, g is the image
    I_gray_copy = g.copy()

    x,y = h.shape
    xl = int(x/2)
    yl = int(y/2)
    for i in range(xl,len(g[:,1])-xl):
        for j in range(yl, len(g[i,:])-yl):

            f = g[i-xl:i+(xl+1), j-yl:j+(yl+1)] #FIXME

            total = h*f
            I_gray_copy[i][j] = sum(sum(total))
    return I_gray_copy

def gauss_kernal(size, var):
    kernel = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(size):
            kernel[i][j] = mt.exp( -((i - (size-1)/2)**2 + (j - (size-1)/2)**2 )/(2*var*var))
    kernel = kernel / kernel.sum()
    return kernel

def harris_response(img):
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gauss = gauss_kernal(3,2)
    #calculate the harris response using sobel operator and gaussian kernel

    Iu = convolve(img,sobel)
    Iv = convolve(img,sobel.transpose())

    Iuu = convolve(Iu*Iu,gauss)
    Ivv = convolve(Iv*Iv,gauss)
    Iuv = convolve(Iu*Iv,gauss)

    H = (Iuu*Ivv - Iuv*Iuv)/(Iuu + Ivv + .0000000001)

    return H

def getmaxima (H,threshold,localSearchWidth = 5):
    maxima = []

    p = localSearchWidth

    width,height = H.shape
    for i in range(int(p/2)+1,width-int(p/2)+1,p):
        for j in range(int(p/2)+1,height-int(p/2)+1,p):
            if H[i,j] < threshold:
                continue
            else:
                localMax = [0,0,0]
                for x in range(i-int(p/2),i+int(p/2)):
                    for y in range(j-int(p/2),j+int(p/2)):
                        if(H[x][y] > localMax[2]):
                            localMax = [x,y, H[x][y]]
                maxima.append(localMax)
    return maxima

def nonmaxsup(H,n=100,c=.9):

    mindistance = []
    threshold = np.mean(H) + np.std(H)
    maxima = np.array(getmaxima(H,threshold))

    x = 0
    y = 1
    z = 2
    for row in maxima:
        min = np.inf
        for row1 in maxima:
            if (row[z] < c*row1[z]):
                dist = np.sqrt((row[x]-row1[x])**2 + (row[y]-row1[y])**2 )
                if (dist < min) and (dist>0):
                    min = dist
                #xmin = row1[x]
                #ymin = row1[y]

        mindistance.append([row[x],row[y],min])
    mindistance.sort(key=lambda x:x[2])

    return mindistance[-n:]

def descriptorExtractor(img, featureList, l = 21):

    def patchFinder(i,j,img,featureList,l):
        descriptor = [0,0,0]
        patch = np.zeros((l,l))
        patchX = 0
        floor = int(l/2)
        ceiling = int(l/2)+1

        #pythons stupid.
        i = int(i)
        j = int(j)

        #find patches, return 0 if out of bounds (this could be improved by not just returning 0)
        for x in range(i-floor,i+ceiling):
            if x < 0 or x >= width:
                return []
            else:
                patchY = 0
                for y in range(j-floor,j+ceiling):
                    if y < 0 or y >= height:
                        return []
                    else:
                        patch[patchX][patchY] = img[x][y]
                        patchY +=1
                patchX +=1
        descriptor[0] = patch
        descriptor[1] = i
        descriptor[2] = j
        return descriptor

    width,height = img.shape

    patches = []
    for point in featureList:
        patch = patchFinder(point[0],point[1],img,featureList,l)
        #Checks to see if patchFinder returned an appropriate patch. Only append if true.
        if(len(patch)> 0):
            patches.append(patch)

    return patches

def sum_squared_error(D1, D2):
    if(D1.shape == D2.shape):
        return (((D1-np.mean(D1))/np.std(D1)) - ((D2-np.mean(D2))/np.std(D2)))**2
    else:
        return np.inf

def get_best_matches(des_I1, des_I2):
    best_matches = []
    best_sse = np.inf
    for i in range(len(des_I1)):
        for j in range(len(des_I2)):
            sse = sum(sum(sum_squared_error(des_I1[i][0], des_I2[j][0])))
            if(sse < best_sse):
                best_sse = sse
                bestMatch = np.array([des_I1[i][1],des_I1[i][2],des_I2[j][1],des_I2[j][2]])
        best_matches.append(np.array([bestMatch,best_sse]))
        best_sse = np.inf
    return best_matches

def get_secondbest_matches(des_I1, des_I2, best_matches):
    secondbest_matches = []
    best_sse = np.inf

    for i in range(len(des_I1)):
        for j in range(len(des_I2)):
            sse = sum(sum(sum_squared_error(des_I1[i][0], des_I2[j][0])))
            if(sse < best_sse and sse != best_matches[i][1]):
                best_sse = sse
                secondBestMatch = np.array([des_I1[i][1],des_I1[i][2],des_I2[j][1],des_I2[j][2]])
        secondbest_matches.append(np.array([secondBestMatch, best_sse]))
        best_sse = np.inf

    return secondbest_matches

def filter_matches(best_matches, secondbest_matches, r=.5):
    filtered_matches = []
    for i in range(len(best_matches)):
        if(best_matches[i][1] < r*secondbest_matches[i][1]):
            filtered_matches.append(best_matches[i][0])
    return filtered_matches

def findHomography(sample):
    A = []

    for match in sample:
        u = match[0]
        v = match[1]
        uP= match[2]
        vP= match[3]
        A.append([0,0,0,-u,-v,-1,vP*u,vP*v,vP])
        A.append([u,v,1,0,0,0,-uP*u,-uP*v,-uP])

    U,Sigma,Vt = np.linalg.svd(A)
    H = Vt[-1]

    H = np.reshape(H, (-1,3))
    return(H)

def RANSAC(number_of_iterations,matches,n,r,d):

    H_best = np.array([[1,0,0],[0,1,0],[0,0,1]])
    list_of_inliers = []

    for i in range(number_of_iterations):
        # 1. Select a random sample of length n from the matches
        samples = []
        for i in range(n):
            idx = random.randint(0,len(matches)-1)
            samples.append(matches.pop(idx))


        # 2. Compute a homography based on these points using the methods given above

        H = findHomography(samples)

        # 3. Apply this homography to the remaining points that were not randomly selected
        predicted = []
        observed = []
        for sample in samples:
            pred = sample[0:2]
            np.append(pred,1)
            predicted.append(pred)

            obs = sample[2:]
            np.append(obs,1)
            observed.append(obs)

        predicted = np.asarray(predicted)

        predicted = (H @ predicted.T).T

        # 4. Compute the residual between observed and predicted feature locations
        inliers = []
        for i in range(len(predicted)):
            pred = predicted[i]
            obs = observed[i]

            #scale
            pred[0]= pred[0]/pred[2]
            pred[1]=pred[1]/pred[2]

            #readability
            u = obs[0]
            v = obs[1]
            uP = pred[0]
            vP = pred[1]

            #calc residual
            resid = np.sqrt((u-uP)**2+(v-vP)**2)
        # 5. Flag predictions that lie within a predefined distance r from observations as inliers
            if(resid < r):
                inliers.append([u,v])

        # 6. If number of inliers is greater than the previous best
        #    and greater than a minimum number of inliers d,
        #    7. update H_best
        #    8. update list_of_inliers

            if(len(inliers) > len(list_of_inliers) and len(inliers) > d):
                list_of_inliers = inliers.copy()
                H_best = H


    return H_best, list_of_inliers
