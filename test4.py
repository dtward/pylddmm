# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:22:04 2017

@author: dtward

2d lddmm

for now this is just a script, not a function

"""

import numpy as np
import matplotlib.pyplot as plt
import lddmm_image_3d as lddmm


plt.close('all')


# image domain
# in general this will be read from an input file
# in general
nx = 50
ny = 50
nz = 50
dx = 1.0
dy = 1.1
dz = 0.9
x0 = 0.5
y0 = 10.3
z0 = 5.3
x = np.arange(nx)*dx + x0
y = np.arange(ny)*dy + y0
z = np.arange(nz)*dz + z0
xx = x[None,None,:] + y[None,:,None]*0 + z[:,None,None]*0
yy = x[None,None,:]*0 + y[None,:,None] + z[:,None,None]*0
zz = x[None,None,:]*0 + y[None,:,None]*0 + z[:,None,None]


# make example images
r = 12.0
# for blurring
rb = 1
x_ = np.arange(-rb,rb+1)
X,Y,Z = np.meshgrid(x_,x_,x_)
K = np.exp(-0.5*(X**2 + Y**2 + Z**2)/(rb*0.5)**2)
K = np.pad(K,((0,xx.shape[0]-K.shape[0]),(0,xx.shape[1]-K.shape[1]),(0,xx.shape[2]-K.shape[2])),'constant')
K = np.roll(K,(-rb,-rb,-rb),axis=(0,1,2))
K = K/np.sum(K)
Khat = np.real(np.fft.fftn(K))

J = []
N = 5
for i in range(N+1):
    mag = 0.2
    J_ = (xx - np.mean(x))**2/(r*(1.0 + np.random.randn()*mag))**2 + (yy - np.mean(y))**2/(r*(1.0 + np.random.randn()*mag))**2 + (zz - np.mean(z))**2/(r*(1.0 + np.random.randn()*mag))**2 <= 1
    J_ = J_ + np.random.randn(*J_.shape)*0.05
    # blur
    J_ = np.real(np.fft.ifftn(np.fft.fftn(J_)*Khat))
    J.append(J_)
I0 = J.pop()
I0 = None

fig = plt.figure()
ax0 = fig.add_subplot(121)
ax0.imshow(J[0][nz/2],extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')
ax1 = fig.add_subplot(122)
try:
    ax1.imshow(J[1][nz/2],extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')
except:
    pass
plt.pause(0.01)

niter = 100
sigmaR = 5.0
sigmaI = 0.1
alpha = 5.0
epsilon = 2.0e-1
epsilonA = 1e-3
nT=5
niter = 1 # this is for the internal lddmm
niterA = 500
nshow = 1

lddmm.lddmm_image_3d_template(x,y,z,J,sigmaI=sigmaI,sigmaR=sigmaR,alpha=alpha,nT=nT,niter=niter,epsilon=epsilon,nshow=nshow,niterA=niterA,epsilonA=epsilonA,vtx=None,vty=None,vtz=None,IA=I0)
# NOTE 
# boundary does not look good, is there something wrong with lambdaI on the boundary?
# some bullshit is going on on the edges

# at the first iteration there's no gradient (if I start at the average)
# starting at the second iteration I see the craziness at the edges