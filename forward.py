import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from scipy.special import hankel1
from scipy import constants

from structure import *

# space configuration
mm = 1e-3
c = 3e8
dx = 4.8 * mm
dy = 4.8 * mm
dz = 4.8 * mm
dt = 1/4 * dx / constants.c
lamb = 74.9*mm
k = 2*np.pi / lamb

# epsilon in Gamma region
eps = np.zeros((250,250))
str1 = Sphere(shape = (250,250,250), center = (125,125,125), R = lamb*3/dx, eps=2, mu=1, smoothing=True)
eps = str1.epsr[:,:,125]
plt.imshow(eps)
plt.show()
del str1

# Green function matrix in Omega region
x = np.arange(94) * dx
y = np.arange(94) * dy
X,Y = np.meshgrid(x,y)
X = X.flatten()
Y = Y.flatten()
Xp = X.reshape(-1,1)
Yp = Y.reshape(-1,1)
r = np.sqrt((X-Xp)**2 + (Y-Yp)**2)
G = 1J/4 * hankel1(0, k * np.sqrt(constants.epsilon_0)  * r)
G = np.nan_to_num(G)

# Green function matrix in Gamma region
x_gamma = np.arange(250) * dx
y_gamma = np.arange(250) * dy
X_g, Y_g = np.meshgrid(x_gamma, y_gamma)
X_g = X_g.flatten().reshape(-1,1)
Y_g = Y_g.flatten().reshape(-1,1)
r_g = np.sqrt((X_g-(X+78*dx))**2 +(Y_g-(Y+78*dx))**2)
H = 1J/4 * hankel1(0, k * np.sqrt(constants.epsilon_0) * r_g)
H = np.nan_to_num(H)

# epsilon in Omega region
Omega = eps[125-47:125+47, 125-47:125+47].copy()
plt.imshow(Omega)
plt.show()
Omega = Omega.flatten()

f = k**2 * constants.epsilon_0 * np.diag((Omega) -1)
A = np.eye(len(f)) - np.matmul(G, f)

#input field
r = np.sqrt(((X_g+(1000/4.8-125)*dx)**2+(Y_g-125*dx)**2).reshape(250,250)[125-47:125+47,125-47:125+47].flatten())
u_in = np.sin(k*r)
plt.imshow(np.real(u_in.reshape(94,94)))
plt.show()

#iterative parameters
delta = 5*10e-7*np.linalg.norm(u_in)
u_prev = u_in
u_prevprev = u_in
t_prev = 0
iter = 1

print("iteration starts")
while iter < 120:
    t = (1 + np.sqrt(1+4*t_prev**2))/2
    mu = (1 - t_prev) / t
    s = (1 - mu)*u_prev + mu*u_prevprev
    g =  np.matmul(np.conj(A.T), (np.matmul(A,s) - u_in))
    gamma = ((np.linalg.norm(g) / np.linalg.norm(np.matmul(A,g))))**2
    if iter % 1 == 0:
        print("now : {}, step : {}".format(np.linalg.norm(g),iter))
    u = s - gamma * g
    u_prev = u
    u_prevprev = u_prev
    t_prev = t
    iter += 1

    if np.linalg.norm(g) < delta:
        break

u_p = np.matmul(H, np.matmul(f,u))
u_p = u_p.reshape(250,250)
u_p[125-47:125+47, 125-47:125+47] =np.matmul(G,np.matmul(f,u)).reshape(94,94)
plt.imshow(np.abs((u_p)), cmap='jet')