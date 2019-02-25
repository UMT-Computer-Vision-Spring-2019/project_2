import numpy as np
import math as mt

def sum_squared_error(D1, D2):
    if(D1.shape == D2.shape):
        return (((D1-np.mean(D1))/np.std(D1)) - ((D2-np.mean(D2))/np.std(D2)))**2
    else:
        return np.inf

def get_best_matches(des_I1, des_I2):
    best_matches = []
    best_sse = np.inf
    for x in range(len(des_I1)):
        for y in range(len(des_I2)):
            sse = sum(sum(sum_squared_error(des_I1[x][0], des_I2[y][0])))
            if(sse < best_sse):
                best_sse = sse

                best_descriptor = des_I2[y]
        best_matches.append(np.array([x,best_descriptor, best_sse]))
        best_sse = np.inf
    return(best_matches)

def get_secondbest_matches(des_I1, des_I2, best_matches):
    secondbest_matches = []
    best_sse = np.inf

    for x in range(len(des_I1)):
        for y in range(len(des_I2)):
            sse = sum(sum(sum_squared_error(des_I1[x][0], des_I2[y][0])))

            if(sse < best_sse and sse != best_matches[x][2]):
                best_sse = sse
                best_descriptor = des_I2[y]
        secondbest_matches.append(np.array([x,best_descriptor, best_sse]))
        best_sse = np.inf
    return secondbest_matches

def filter_matches(best_matches, secondbest_matches, r=.5):
    fMatches = []
    for x in range(len(best_matches)):
        if(best_matches[x][2] < r*secondbest_matches[x][2]):
            fMatches.append(best_matches[x])

    return fMatches

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

def harris_response(img, gmean = 5,var =2):
	sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	gauss = gauss_kernal(gmean,var)
    #calculate the harris response using sobel operator and gaussian kernel

	Iu = convolve(img,sobel)
	Iv = convolve(img,sobel.transpose())

	Iuu = convolve(Iu*Iu,gauss)
	Ivv = convolve(Iv*Iv,gauss)
	Iuv = convolve(Iu*Iv,gauss)

	H = (Iuu*Ivv - Iuv*Iuv)/(Iuu + Ivv + .0000000001)

	return H

def getmaxima (H,threshold,localSearchWidth = 21):
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
        descriptor = [i,j,np.zeros((l,l))]
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
