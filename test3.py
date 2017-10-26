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
nx = 64
ny = 63
nz = 30
dx = 1.0
dy = 1.1
dz = 2.0
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
r = 20.0
I = (xx - np.mean(x))**2/r**2 + (yy - np.mean(y))**2/r**2 + (zz - np.mean(z))**2/r**2 <= 1



J = (xx - np.mean(x))**2/(r*1.2)**2 + (yy - np.mean(y))**2/(r*0.8)**2 + (zz - np.mean(z))**2/(r*1.1)**2 <= 1
J = J + np.random.randn(*J.shape)*0.05
'''
fig = plt.figure()
ax0 = fig.add_subplot(121)
ax0.imshow(I[nz/2],extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')
ax1 = fig.add_subplot(122)
ax1.imshow(J[nz/2],extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')
'''

niter = 100
sigmaR = 5.0
sigmaI = 0.1
alpha = 10.0
epsilon = 1.0e-0

#niter = 5
output = lddmm.lddmm_image_3d(x,y,z,I,x,y,z,J,niter=niter,sigmaR=sigmaR,sigmaI=sigmaI,alpha=alpha,epsilon=epsilon,nshow=1)

# for template estimation I need these things
plt.figure()
plt.imshow(output['lambdaI0'][nz/2],cmap='gray',interpolation='none')
plt.colorbar()

plt.figure()
plt.imshow(output['detjacphi10inv'][nz/2],cmap='gray',interpolation='none')
plt.colorbar()

# looking at detjac there seems to be a problem with left and right boundaries
# it spans the first 2 rows