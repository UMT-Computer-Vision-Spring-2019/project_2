import numpy as np
import matplotlib.pyplot as plt

def convolve(g,h):
  gx = g.shape[0]
  gy = g.shape[1]

  hx = h.shape[0]
  hy = h.shape[1]
  padx = hx/2
  pady = hy/2

  res = np.matrix(gx - padx * 2, gy - pady * 2)
  print(gx, gy, res.shape)  
  for x in range (padx, gx - padx):
    for y in range(pady, gy - pady):
      g_local = g[x-padx:x+padx+1,y-pady:y+pady+1]
      print(g_local)
      np.sum(np.multiply(g_local, h))
  return 0

img_color = plt.imread('stuff.jpeg')
I_gray = img_color.mean(axis=2)
Su = np.matrix(
[[-1, 0, 1],
[-2, 0, 2],
[-1,0,1]])

plt.imshow(convolve(I_gray, Su))
