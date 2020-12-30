import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from scipy.special import hankel1
from scipy import constants

from structure import *

um = 1e-6
dx = 4.8 * um
dy = 4.8 * um
dz = 4.8 * um
dt = 1/4 * dx / constants.c
lamb = 74.9*um
k = 2*np.pi / lamb

# space configuration
eps = np.zeros((250,250))
str1 = Sphere(shape = (250,250,250), center = (125,125,125), R = lamb*3/dx, eps=10, mu=1)
eps = str1.epsr[:,:,125]
plt.imshow(eps)
plt.show()
del str1

# Omega region configuration
x = np.arange(94) * dx
y = np.arange(94) * dy
X,Y = np.meshgrid(x,y)
X = X.flatten()
Y = Y.flatten()
Xp = X.reshape(-1,1)
Yp = Y.reshape(-1,1)
r = np.sqrt((X-Xp)**2 + (Y-Yp)**2)
G = 1/4 * hankel1(0, k*r)
print("calculating Omega region green function")
G = np.nan_to_num(G)
plt.imshow(np.abs(G))

# Gamma region configuration

x_gamma = np.arange(250) * dx
y_gamma = np.arange(250) * dy
X_g, Y_g = np.meshgrid(x_gamma, y_gamma)
X_g = X_g.flatten().reshape(-1,1)
Y_g = Y_g.flatten().reshape(-1,1)
r_g = np.sqrt((X_g-(X+78*dx))**2 +(Y_g-(Y+78*dx))**2)
print("calculating Gamma region green function")
H = 1/4 * hankel1(0, k*r_g)
H = np.nan_to_num(H)
plt.imshow(np.abs(H))


# Omega region scatterer.

Omega = eps[125-47:125+47, 125-47:125+47].copy()
# plt.imshow(Omega)
# plt.show()
Omega = Omega.flatten()


# Omega region function

f = k**2 * np.diag((Omega)**2 -1)
A = np.eye(len(f)) - np.matmul(G, f)


# input function

u_in = np.exp(1J * k*(np.sqrt((X_g-(1000/4.8+125)*dx)**2+(Y_g-125*dx)**2))).reshape(250,250)[125-47:125+47,125-47:125+47].flatten()
# u_in = np.ones((250,250)).reshape(250,250)[125-47:125+47,125-47:125+47].flatten()
# plt.imshow(np.real(u_in.reshape(94,94)))
# plt.show()


# iterative parameter configuration
delta = 5*10e-7*np.linalg.norm(u_in)
u_prev = u_in
u_prevprev = u_in
t_prev = 0
iter = 1

#iteration
print("iteration starts")
while iter < 120:
    t = (1 + np.sqrt(1+4*t_prev**2))/2
    mu = (1 - t_prev) / t
    s = (1 - mu)*u_prev + mu*u_prevprev
    g = np.matmul(np.conj(A.T), (np.matmul(A,s) - u_in))
    gamma = ((np.linalg.norm(g) / np.linalg.norm(np.matmul(A,g))))**2
    if iter % 20 == 0:
        print("now : {}, step : {}".format(gamma * np.linalg.norm(g),iter))
    if gamma * np.linalg.norm(g) < delta:
        break
    u = s - gamma * g
    u_prev = u
    u_prevprev = u_prev
    t_prev = t
    iter += 1

# total field
u_p = np.matmul(H, u*(Omega**2-1))
u_p = u_p.reshape(250,250)
u_p[125-47:125+47, 125-47:125+47] =np.matmul(G, u*(Omega**2-1)).reshape(94,94)
plt.imshow(np.abs(u_p), cmap='jet')
plt.show()
plt.imshow(np.abs(np.exp(1J * k*(np.sqrt((X_g-(1000/4.8+125)*dx)**2+(Y_g-125*dx)**2)).reshape(250,250))))