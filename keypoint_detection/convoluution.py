import numpy as np
import matplotlib.pyplot as plt

def convolve_no_edge(g,h):
  gx = g.shape[0]
  gy = g.shape[1]

  hx = h.shape[0]
  hy = h.shape[1]
  padx = hx/2
  pady = hy/2
  res = np.zeros((gx - (padx - 1), gy - (pady - 1)))
  for x in range (padx, gx - padx):
    for y in range(pady, gy - pady):
      g_local = g[x-padx:x+padx+1,y-pady:y+pady+1]
      res_pixel = np.sum(np.multiply(g_local, h))
      res[x-padx,y-pady] = res_pixel
  return res
  
def convolve_fill(g,h, fill_val):
  gx = g.shape[0]
  gy = g.shape[1]

  hx = h.shape[0]
  hy = h.shape[1]
  padx = hx/2
  pady = hy/2

  res = np.zeros((gx, gy))
  for x in range (0, gx):
    for y in range(0, gy):
      g_local = np.array([])
      for j in range(x - padx, x + padx + 1):
        for k in range(y - pady, y + pady + 1):
          if (j < 0 or j >= gx or k < 0 or k >= gy):
            g_local = np.append(g_local, fill_val)
          else:
            g_local = np.append(g_local, g[j,k])
      g_local = g_local.reshape((3,3))    
      res_pixel = np.sum(np.multiply(g_local, h))
      res[x-padx,y-pady] = res_pixel
  return res
  
def convolve_nearest_neighbors(g, h):
  gx = g.shape[0]
  gy = g.shape[1]

  hx = h.shape[0]
  hy = h.shape[1]
  padx = hx/2
  pady = hy/2

  res = np.zeros((gx, gy))
  for x in range (0, gx):
    for y in range(0, gy):
      g_local = np.array([])
      for j in range(x - padx, x + padx + 1):
        for k in range(y - pady, y + pady + 1):
          if (j < 0 or j >= gx or k < 0 or k >= gy):
            j_copy = j
            k_copy = k
            while ((j_copy < 0 or j_copy >= gx) or (k_copy < 0 or k_copy >= gy)):
              if (j_copy < 0):
                j_copy+=1
              elif (j_copy >= gx):
                j_copy-=1
              if (k_copy < 0):
                k_copy+=1
              elif (k_copy >= gy):
                k_copy-=1    
            g_local = np.append(g_local, g[j_copy, k_copy])
          else:
            g_local = np.append(g_local, g[j,k])
      g_local = g_local.reshape((3,3)) 
      res_pixel = np.sum(np.multiply(g_local, h))
      print(res_pixel)
      res[x-padx,y-pady] = res_pixel
  return res
'''  
img_color = plt.imread('stuff.jpeg')
I_gray = img_color.mean(axis=2)
Su = np.matrix(
[[-1, 0, 1],
[-2, 0, 2],
[-1,0,1]])

filter = convolve_nearest_neighbors(I_gray, Su.T)
plt.imshow(filter, cmap='gray')
plt.show()
'''
