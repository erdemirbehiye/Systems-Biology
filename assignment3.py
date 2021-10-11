from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from random import random

# generates 101-element array from 0 to 1.01 in 0.01 increments
x = np.arange(0, 1.01, 0.01)

# creates 2-D coordinate arrays, given one-dimensional coordinate arrays x1, x2
X1, X2 = np.meshgrid(x, x)

# random weights and tresholds
w = np.random.normal(loc=0.0, scale=1.0, size=9)
T = np.random.normal(loc=0.0, scale=1.0, size=4)

# multi-layer perceptron network
Y1=  (w[0] * X1 + w[1] * X2 >= T[0])
Y2 = (w[2] * X1 + w[3] * X2 >= T[1])
Y3 = (w[4] * X1 + w[5] * X2 >= T[2])
Z = (w[6] * Y1 + w[7] * Y2 + w[8] * Y3 >= T[3])

# desired response
border = np.logical_or(np.logical_and(X1>=0.3,X1<=0.6),np.logical_and(X2>=0.2,X2<=0.7))


fig=plt.figure(figsize=(20,6))
ax=fig.add_subplot(1,2,1)
h=ax.imshow(X1,origin='lower',extent=[0,1,0,1],cmap=matplotlib.cm.jet)
ax.set_xlabel('$[X_1]$')
ax.set_ylabel('$[X_2]$')
ax.set_title('$[X_1]$')
fig.colorbar(h)


ax=fig.add_subplot(1,2,2)
h=ax.imshow(X2,origin='lower',extent=[0,1,0,1],cmap=matplotlib.cm.jet)
ax.set_xlabel('$[X_1]$')
ax.set_ylabel('$[X_2]$')
ax.set_title('$[X_2]$')
fig.colorbar(h)


fig, ax = plt.subplots()
plt.imshow(border,cmap='OrRd',interpolation='None',extent=[0,1,0,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(linestyle='--')
plt.show()

# initial response characteristic of the system 
err=np.sum(border != Z)
print(100*err/(len(border)**2))



#initialization for the good variables
w_new=w
T_new=T
Z_new=Z

i =0

#iterate for the best scenario
while i<100000:
  
    w =np.random.normal(loc=w_new,scale=1.0,size=9)
    T =np.random.normal(loc=T_new,scale=1.0,size=4)
    Y1=(w[0] * X1 + w[1] * X2 >= T[0])
    Y2=(w[2] * X1 + w[3] * X2 >= T[1])
    Y3=(w[4] * X1 + w[5] * X2 >= T[2])
    Z=(w[6] * Y1 + w[7] * Y2 + w[8] * Y3 >= T[3])
    
    #find the difference
    err_best=np.sum(border != Z)
    
    #change the condition
    if err_best < err:
        w_new=w
        T_new=T
        Z_new=Z
        err=err_best
    
    i=i+1


fig, ax = plt.subplots()
plt.imshow(border != Z_new,extent=[0,1,0,1], cmap='OrRd')
plt.title('', fontsize=8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(linestyle='--')
plt.show()

print(100*err/(len(border)**2))