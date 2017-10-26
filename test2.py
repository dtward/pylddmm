# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:22:04 2017

@author: dtward

2d lddmm

for now this is just a script, not a function

"""

import numpy as np
import matplotlib.pyplot as plt
import lddmm_image_2d as lddmm


plt.close('all')


# image domain
# in general this will be read from an input file
# in general
nx = 64
ny = 63
dx = 1.0
dy = 1.1
x0 = 0.5
y0 = 10.3
x = np.arange(nx)*dx + x0
y = np.arange(ny)*dy + y0
xx,yy = np.meshgrid(x,y)


# make example images
r = 20.0
I = (xx - np.mean(x))**2/r**2 + (yy - np.mean(y))**2/r**2 <= 1
fig = plt.figure()
ax0 = fig.add_subplot(121)
ax0.imshow(I,extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')


J = (xx - np.mean(x))**2/(r*1.2)**2 + (yy - np.mean(y))**2/(r*0.8)**2 <= 1
J = J + np.random.randn(*J.shape)*0.1
ax1 = fig.add_subplot(122)
ax1.imshow(J,extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')


niter = 1000
sigmaR = 50.0
sigmaI = 0.1
alpha = 5

lddmm.lddmm_image_2d(x,y,I,x,y,J,niter=niter,sigmaR=sigmaR,sigmaI=sigmaI,alpha=5)