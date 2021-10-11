import numpy as np
import math
import matplotlib.pyplot as plt

dt = 0.001
t = np.arange(0,5,dt)

X = np.ndarray((len(t)))
Y = np.ndarray((len(t)))
E = np.ndarray((len(t)))
XY = np.ndarray((len(t)))
XE = np.ndarray((len(t)))
YE = np.ndarray((len(t)))

k1 = 0.5
k2 = 0.8
k3 = 0.8
k4 = 1.75
k5 = 1.2
k6 = 1.5
k7 = 2.5
k8 = 2.2

X_i = 1.0
Y_i = 1.0
E_i = 0.0
XY_i = 0.0
XE_i = 0.0
YE_i = 0.0

cind = 0

for ct in t:
  dXdt = k2*XY_i + k4*XE_i - k1*X_i*Y_i - k3*X_i*E_i - k8*YE_i*X_i
  dYdt = k2*XY_i + k6*YE_i - k1*X_i*Y_i - k5*Y_i*E_i - k7*XE_i*Y_i
  dEdt = k4*XE_i + k6*YE_i + k7*XE_i*Y_i + k8*YE_i*X_i - k3*X_i*E_i - k5*Y_i*E_i
  dXYdt = k1*X_i*Y_i + k7*XE_i*Y_i + k8*YE_i*X_i - k2*XY_i
  dXEdt = k3*X_i*E_i - k4*XE_i - k7*XE_i*Y_i
  dYEdt = k5*Y_i*E_i - k6*YE_i - k8*YE_i*X_i

  X_n = X_i + dt * dXdt
  Y_n = Y_i + dt * dYdt
  E_n = E_i + dt * dEdt
  XY_n = XY_i + dt * dXYdt
  XE_n = XE_i + dt * dXEdt
  YE_n = YE_i + dt * dYEdt


  X[cind] = X_n
  Y[cind] = Y_n
  E[cind] = E_n
  XY[cind] = XY_n
  XE[cind] = XE_n
  YE[cind] = YE_n

  """G1[cind] = X_n + XY_n + XE_n
  G2[cind] = Y_n + XY_n + YE_n
  G3[cind] = E_n + XE_n + YE_n"""


  X_i = X_n
  Y_i = Y_n
  E_i = E_n
  XY_i = XY_n
  XE_i = XE_n
  YE_i = YE_n

  cind +=1

plt.figure(dpi=100)
plt.plot(t, X, label='X')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

plt.figure(dpi=100)
plt.plot(t, Y, label='Y')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

plt.figure(dpi=100)
plt.plot(t, E, label='E')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

plt.figure(dpi=100)
plt.plot(t, XY, label='XY')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

plt.figure(dpi=100)
plt.plot(t, XE, label='XE')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

plt.figure(dpi=100)
plt.plot(t, YE, label='YE')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()

"""plt.figure(dpi=100)
plt.plot(t, G1, label='G1')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()
"""
"""plt.figure(dpi=100)
plt.plot(t, G2, label='G2')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()


plt.figure(dpi=100)
plt.plot(t, G3, label='G3')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.show()
"""
"""print(G3)"""