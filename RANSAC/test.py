import numpy as np
import matplotlib.pyplot as plt


X = np.array([0,0,1])

H = np.random.rand(3,3)
#H/= H[2,2]

Xprime = (H.dot(X.T)).T
Xprime/=Xprime[:,2][:,np.newaxis]

plt.plot(X[:,0],X[:,1],'g-')
plt.plot(Xprime[:,0],Xprime[:,1],'b-')
plt.show()
