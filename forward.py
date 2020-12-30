import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from scipy.special import hankel1
from scipy import constants

from structure import *

mm = 1e-3
c = 3e8
dx = 4.8 * mm
dy = 4.8 * mm
dz = 4.8 * mm
dt = 1/4 * dx / constants.c
lamb = 74.9*mm
k = 2*np.pi / lamb

# space configuration
eps = np.zeros((250,250))
str1 = Sphere(shape = (250,250,250), center = (125,125,125), R = lamb*3/dx, eps=2, mu=1)
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
print("calculating Omega region green function")
G = 1J/4 * hankel1(0, k*r)
G = np.nan_to_num(G)

# Gamma region configuration
x_gamma = np.arange(250) * dx
y_gamma = np.arange(250) * dy
X_g, Y_g = np.meshgrid(x_gamma, y_gamma)
X_g = X_g.flatten().reshape(-1,1)
Y_g = Y_g.flatten().reshape(-1,1)
r_g = np.sqrt(((X+78*dx)-X_g)**2 +((Y+78*dx)-Y_g)**2)
print("calculating Gamma region green function")
H = 1J/4 * hankel1(0, k*r_g)
H = np.nan_to_num(H)


# Omega region scatterer.

Omega = eps[125-47:125+47, 125-47:125+47].copy()
# plt.imshow(Omega)
# plt.show()
Omega = Omega.flatten()

# Omega region function
f = k**2*constants.epsilon_0* np.diag(Omega -1)
A = np.eye(len(f)) - np.matmul(G, f)

# input function
rad = np.sqrt((X_g-1000*mm-125*dx)**2+(Y_g-125*dx)**2).reshape(250,250)
u_input = np.exp(1J*k*rad)
u_in = u_input[125-47:125+47,125-47:125+47].flatten()
# u_in = np.ones((250,250)).reshape(250,250)[125-47:125+47,125-47:125+47].flatten()
plt.imshow(np.real(u_in.reshape(94,94)))
plt.show()


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
    gamma = ((np.linalg.norm(g) / np.linalg.norm(np.matmul(A,g))))
    if iter % 1 == 0:
        print("now : {}, step : {}".format(np.linalg.norm(g),iter))
    if np.linalg.norm(g) < delta:
        break
    u = s - gamma * g
    u_prev = u
    u_prevprev = u_prev
    t_prev = t
    iter += 1

# total field
u_p = np.matmul(H, np.matmul(f,u))
u_p = u_p.reshape(250,250)
u_p[125-47:125+47, 125-47:125+47] =np.matmul(G, np.matmul(f,u)).reshape(94,94)
plt.imshow(np.abs(u_p+u_input), cmap='jet')
plt.show()
