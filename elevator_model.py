import numpy as np
import matplotlib.pyplot as plt

floor_distance=3.0
floor_choice=1.0
t_end=floor_choice*20
speed=20.0

dt = 0.001
t = np.arange(0,t_end,dt)

X = np.ndarray((len(t)))
H = np.ndarray((len(t)))
D = np.ndarray((len(t)))
Y = np.ndarray((len(t)))
A = np.ndarray((len(t)))
S = np.ndarray((len(t)))
F = np.ndarray((len(t)))
S_floor=np.ndarray((len(t)))
X_v = np.ndarray((len(t)))
Y_v = np.ndarray((len(t)))
R_v = np.ndarray((len(t)))
Z_v = np.ndarray((len(t)))
Y_dist = np.ndarray((len(t)))
Z_dist = np.ndarray((len(t)))

alpha_X=1.0
alpha_Y=1.0
alpha_A=1.0
alpha_S=1.0
alpha_F=1.0
alpha_S_floor=1.0
alpha_X_v=1.0
alpha_R_v=1.0
alpha_Y_v=1.0
alpha_Z_v=1.0
alpha_Y_dist=0.25
alpha_Z_dist=0.25

beta_X=1.0
beta_Y=1.0
beta_A=1.0
beta_S=1.0
beta_F=1.0
beta_S_floor=1.0
beta_X_v=speed
beta_Y_v=speed
beta_R_v=speed
beta_Z_v=speed
beta_Y_dist=floor_distance*alpha_Y_dist*floor_choice
beta_Z_dist=3.4*floor_distance*alpha_Y_dist*floor_choice

c_X = 0.0
c_Y = 0.0
c_A = 0.0
c_S = 0.0
c_F = 0.0
c_S_floor=0.0
c_X_v = 0.0
c_Y_v = 0.0
c_R_v = 0.0
c_Z_v = 0.0
c_Y_dist = 0.0
c_Y_dist = 0.0
c_Z_dist = 0.0
c_H=0.0

n_F=10
n_F2Y=10
n_YdZd=10
n_Sf_Yd=10
n_X2S_floor=10
n_Y2A=10
n_Rv2Yv=10
n_Rv2Zv=10
n_D2Y=10
n_S_floor2X=10
n_Yv2Zv=10
n_X2Y= 10

k_F=0.5
k_F2Y=0.5
k_YdZd=0.5
k_Sf_Yd=0.5
k_X2S_floor=0.5
k_Y2A=0.5
k_D2Y=0.5
k_S_floor2X=0.5
k_X2Y = 0.5
k_Rv2Yv=10
k_Rv2Zv=0.5
k_Yv2Zv=0.5

cind = 0
for ct in t:
  c_X = 1.0
  c_H = 1.0
  #c_H = 1.0 * ((ct >= 1.5) & (ct <=2)).astype(float) + 1*((ct >= 3) & (ct <= t_end)).astype(float) #elevator asymmetric behavior
  c_D=c_X*c_H
  dYdt=beta_Y*(c_D**n_D2Y)/(k_D2Y**n_D2Y+c_D**n_D2Y)- alpha_Y * c_Y
  dAdt = beta_A*(c_Y**n_Y2A)/(k_Y2A**n_Y2A+c_Y**n_Y2A)- alpha_A * c_A
  c_S=0.0 #stop button
  dFdt = beta_F *(1/((1+(c_S/k_F)**n_F)))*(c_X**n_F/(k_F**n_F+c_X**n_F))- alpha_F * c_F
  dS_floordt = beta_S_floor*(c_X**n_X2S_floor)/(k_X2S_floor**n_X2S_floor+c_X**n_X2S_floor)- alpha_S_floor * c_S_floor
  dX_vdt = beta_X_v*(c_S_floor**n_S_floor2X)/(k_S_floor2X**n_S_floor2X+c_S_floor**n_S_floor2X)- alpha_X_v * c_X_v
  dY_vdt = beta_Y_v*(c_X_v**n_X2Y)/(k_X2Y**n_X2Y+c_X_v**n_X2Y)*(c_F**n_F2Y/(k_F2Y**n_F2Y+c_F**n_F2Y))- alpha_Y_v * c_Y_v
  dR_vdt = beta_R_v*(c_Y_v**n_Rv2Yv)/(k_Rv2Yv**n_Rv2Yv+c_Y_v**n_Rv2Yv)- alpha_R_v * c_R_v
  dZ_vdt = beta_Z_v * (c_Y_v**n_Yv2Zv/(k_Yv2Zv**n_Yv2Zv + c_Y_v**n_Yv2Zv))*(1/(1 + ((c_R_v/k_Rv2Zv)*((ct >= t_end-5) & (ct <= t_end)).astype(float))**n_Rv2Zv)) - alpha_Z_v * c_Z_v
  dY_distdt = beta_Y_dist*(c_S_floor**n_Sf_Yd)/(k_Sf_Yd**n_Sf_Yd + c_S_floor**n_Sf_Yd)- alpha_Y_dist * c_Y_dist
  dZ_distdt = beta_Z_dist*((1)/(1+(c_Y_dist/k_YdZd)**n_YdZd))- alpha_Z_dist * c_Z_dist


  nY = c_Y + dt * dYdt
  nA = c_A + dt * dAdt
  nF = c_F + dt * dFdt
  n_S_floor = c_S_floor + dt * dS_floordt
  n_X_v = c_X_v + dt * dX_vdt
  n_Y_v = c_Y_v + dt * dY_vdt
  n_R_v = c_R_v + dt * dR_vdt
  n_Z_v = c_Z_v + dt * dZ_vdt
  n_Y_dist = c_Y_dist + dt * dY_distdt
  n_Z_dist = c_Z_dist + dt * dZ_distdt


  X[cind] = c_X
  H[cind] = c_H
  D[cind] = c_D 
  Y[cind] = nY
  A[cind] = nA
  S[cind] = c_S
  F[cind] = nF
  S_floor[cind] =  n_S_floor
  X_v[cind] = n_X_v
  Y_v[cind] = n_Y_v
  R_v[cind] = n_R_v
  Z_v[cind] = n_Z_v
  Y_dist[cind] = n_Y_dist
  Z_dist[cind] = n_Z_dist

  c_Y = nY
  c_A = nA
  c_F = nF
  c_S_floor =  n_S_floor
  c_X_v = n_X_v
  c_Y_v = n_Y_v
  c_R_v = n_R_v
  c_Z_v = n_Z_v
  c_Y_dist = n_Y_dist
  c_Z_dist = n_Z_dist

  cind +=1
  
  

plt.subplot(7, 2, 1)
plt.plot(t,X)
plt.ylabel('Signal "X"')
plt.xlabel('time')
plt.grid()
plt.show()

plt.subplot(7, 2, 2)
plt.plot(t,H)
plt.ylabel('Signal "H"')
plt.xlabel('time')
plt.grid()
plt.show()

plt.subplot(7, 2, 3)
plt.plot(t,D)
plt.xlabel('time')
plt.ylabel('Signal "D"')
plt.grid()
plt.show()
 
plt.subplot(7, 2, 4)
plt.plot(t,Y)
plt.xlabel('time')
plt.ylabel('Rate of Y')
plt.grid()
plt.show()

plt.subplot(7, 2, 5)
plt.plot(t,A)
plt.xlabel('time')
plt.ylabel('Rate of A')
plt.grid()
plt.show()

plt.subplot(7, 2, 6)
plt.plot(t,S)
plt.xlabel('time')
plt.ylabel('Rate of S')
plt.grid()
plt.show()

plt.subplot(7, 2, 7)
plt.plot(t,F)
plt.xlabel('time')
plt.ylabel('Rate of F')
plt.grid()
plt.show()

plt.subplot(7, 2, 8)
plt.plot(t,S_floor)
plt.xlabel('time')
plt.ylabel('Rate of S_floor')
plt.grid()
plt.show()

plt.subplot(7, 2, 9)
plt.plot(t,X_v)
plt.xlabel('time')
plt.ylabel('Rate of X_v')
plt.grid()
plt.show()

plt.subplot(7, 2, 10)
plt.plot(t,Y_v)
plt.xlabel('time')
plt.ylabel('Rate of Y_v')
plt.grid()
plt.show()

plt.subplot(7, 2, 11)
plt.plot(t,R_v)
plt.xlabel('time')
plt.ylabel('Rate of R_v')
plt.grid()
plt.show()

plt.subplot(7, 2, 12)
plt.plot(t,Z_v)
plt.xlabel('time')
plt.ylabel('Rate of Z_v')
#plt.axis([0, 1, 0, 20*floor_choice ])
plt.grid()
plt.show()

plt.subplot(7, 2, 13)
plt.plot(t,Y_dist)
plt.xlabel('time')
plt.ylabel('Rate of Y_dist')
plt.grid()
plt.show()

plt.subplot(7, 2, 14)
plt.plot(t,Z_dist)
plt.xlabel('time')
plt.ylabel('Rate of Z_dist')
plt.grid()
plt.show()
