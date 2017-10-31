# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:22:04 2017

@author: dtward

3d lddmm

for now this is just a script, not a function


test 5 will use real brain data!

"""

import numpy as np
import matplotlib.pyplot as plt
import lddmm_image_3d as lddmm
import glob

plt.close('all')


filenames = glob.glob('brains/*.png')
#filenames = filenames[:4]
J = []
for filename in filenames:
    J_ = plt.imread(filename)
    J__ = np.concatenate( (J_[None,:,:],J_[None,:,:],J_[None,:,:]),axis=0)
    J.append(J__)

nz = J[0].shape[0]
ny = J[0].shape[1]
nx = J[0].shape[2]
dx = 1.0
dy = 1.0
dz = 10.0
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z = np.arange(0,nz)*dz
N = len(J)


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
sigmaR = 100.0 
sigmaI = 0.1
alpha = 25.0
epsilon = 2.0e0
epsilonA = 5e-4/N
nT=4
niter = 1 # this is for the internal lddmm
niterA = 500
nshow = 1
#lddmm.lddmm_image_3d(x,y,z,J[1],x,y,z,J[2],sigmaI=sigmaI,sigmaR=sigmaR,alpha=alpha,nT=nT,niter=100,epsilon=epsilon,nshow=nshow,vtx=None,vty=None,vtz=None)
#raise Exception

# note, my memory has been going up to about 25% per process before it gets garbage collected
# plus the main thread might have an extra 25%
# I fixed the memory issue by only initializing the pool once, before garbage is accumulated
npool = 4
# or None
sigmaR = 50.0
lddmm.lddmm_image_3d_template(x,y,z,J,sigmaI=sigmaI,sigmaR=sigmaR,alpha=alpha,nT=nT,niter=niter,epsilon=epsilon,nshow=nshow,niterA=niterA,epsilonA=epsilonA,vtx=None,vty=None,vtz=None,IA=I0,npool=npool)

# well I'm not sure I'm getting the same result as without multiprocessing
# I'll have to check carefully