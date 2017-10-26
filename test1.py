# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:22:04 2017

@author: dtward

2d lddmm

for now this is just a script, not a function

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
plt.close('all')


# some parameters
nT = 5
niter = 1000
sigmaI = 0.1
sigmaR = 50.0
epsilon = 0.1 # step size
alpha = 5.0 # smoothness in same units as dx
nshow = 10 # draw a picture every nshow iterations of optimization, 0 for don't show
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

# we also need frequency domain
# and lowpass and highpass operator
fx = np.arange(nx)/dx/nx # note divide by dx first because it is float
fy = np.arange(ny)/dy/ny
fxx,fyy = np.meshgrid(fx,fy)
# AI[i,j] = I[i,j] - alpha^2( (I[i+1,j] - 2I[i,j] + I[i-1,j])/dx^2 + (I[i,j+1] - 2I[i,j] + I[i,j-1])/dy^2  )
Lhat = 1.0 - alpha**2*((-2.0 + 2.0*np.cos(2*np.pi*dx*fxx))/dx**2 + (-2.0 + 2.0*np.cos(2*np.pi*dy*fyy))/dy**2   )
Ahat = Lhat**2
Khat = 1.0/Ahat

# make example images
r = 20.0
I = (xx - np.mean(x))**2/r**2 + (yy - np.mean(y))**2/r**2 <= 1
fig = plt.figure()
ax0 = fig.add_subplot(121)
ax0.imshow(I,extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')


J = (xx - np.mean(x))**2/(r*1.2)**2 + (yy - np.mean(y))**2/(r*0.8)**2 <= 1
ax1 = fig.add_subplot(122)
ax1.imshow(J,extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')


# now initialize our transformation
# note that the transformation will match the sampling of J
# these are the points we will need to evaluate our deformed image at in order to calculate error
dt = 1.0/nT
# I need to save v as a function of t to optimize over
vtx = np.zeros((nT,ny,nx))
vty = np.zeros((nT,ny,nx))

# I need to save I as a function of t because I will need its gradient
# note that in certain sparse sampling schemes it will be better to calculate 
# the gradient first, and then deform it (push forward of vector field)
It = np.zeros((nT+1,ny,nx))
It[0] = np.array(I)


# interpolation will be used for deforming an image, 
# and constructing diffeomorphisms
# note when deforming two things with the same transformation, we want to do 
# both at the same time
interpolator = interp.RegularGridInterpolator((y,x),I,method='linear',bounds_error=False,fill_value=0.0)

# start a gradient descent optimzation loop
for iteration in xrange(niter):    
    # flow forward
    reg_cost = 0.0 # initialize flow energy    
    phiinvx = np.array(xx) # initialize diffeomorphisms
    phiinvy = np.array(yy)
    for t in xrange(nT):
        # update the diffeomorphism
        # here we say phi(x) - identity(x) should have 0 boundary conditions
        # we evaluate it at x - v(x)*dt, then we add back identity(x - v(x)*dt)
        evalx  = xx-vtx[t]*dt
        evaly = yy-vty[t]*dt
        values = np.empty((ny,nx,2))# to evaluate both at once, I need this order
        values[:,:,0]  = phiinvx-xx
        values[:,:,1] = phiinvy-yy        
        interpolator.values = values
        out = interpolator((evaly,evalx))
        phiinvx = out[:,:,0]+xx-vtx[t]*dt
        phiinvy = out[:,:,1]+yy-vty[t]*dt
        # okay now deform the image
        interpolator.values = I
        It[t+1] = interpolator((phiinvy,phiinvx))
        
        # energy of the flow
        Avx = np.real(np.fft.ifft2(np.fft.fft2(vtx[t])*Ahat))
        Avy = np.real(np.fft.ifft2(np.fft.fft2(vty[t])*Ahat))
        reg_cost += 0.5*np.sum(vtx[t]*Avx + vty[t]*Avy)/sigmaR**2
        
    # now we get the cost
    match_cost = 0.5*np.sum((It[-1] - J)**2)/sigmaI**2
    energy = reg_cost + match_cost
    print('iteration {}, energy {} (reg {} + match {}))'.format(iteration, energy, reg_cost, match_cost))
    
    # initialize the gradient of the matching term
    lambdaI = (It[-1] - J)/sigmaI**2
    
    # make a nice picture
    if nshow and not iteration%nshow:
        plt.clf()
        ax0 = fig.add_subplot(121)
        ax0.imshow(It[t+1],extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')
        ax1 = fig.add_subplot(122)
        h = ax1.imshow(lambdaI,extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',cmap='gray')
        plt.colorbar(mappable=h)
        plt.pause(1.0e-10) # draw it now
    
    # now we will backpropogate the gradient backward in time
    phiinvx = np.array(xx)
    phiinvy = np.array(yy)
    for t in range(nT-1,-1,-1):
        # construct the diffeomorphism
        evalx = xx + vtx[t]*dt
        evaly = yy + vty[t]*dt
        values = np.empty((ny,nx,2))# to evaluate both at once, I need this order
        values[:,:,0] = phiinvx-xx
        values[:,:,1] = phiinvy-yy
        interpolator.values = values
        out = interpolator((evaly,evalx))
        phiinvx = out[:,:,0]+xx+vtx[t]*dt
        phiinvy = out[:,:,1]+yy+vty[t]*dt
        
        # we need determinant of jacobian, use centered difference
        phiinvxx = (np.roll(phiinvx,-1,axis=1) - np.roll(phiinvx,1,axis=1))/(2.0*dx)
        phiinvxy = (np.roll(phiinvx,-1,axis=0) - np.roll(phiinvx,1,axis=0))/(2.0*dy)
        phiinvyx = (np.roll(phiinvy,-1,axis=1) - np.roll(phiinvy,1,axis=1))/(2.0*dx)
        phiinvyy = (np.roll(phiinvy,-1,axis=0) - np.roll(phiinvy,1,axis=0))/(2.0*dy)
        # careful boundary conditions
        phiinvxx[:,0] = phiinvxx[:,1]
        phiinvxx[:,-1] = phiinvxx[:,-2]
        phiinvxy[0,:] = phiinvxy[1,:]
        phiinvxy[-1,:] = phiinvxy[-2,:]
        phiinvyx[:,0] = phiinvyx[:,1]
        phiinvyx[:,-1] = phiinvyx[:,-2]
        phiinvyy[0,:] = phiinvyy[1,:]
        phiinvyy[-1,:] = phiinvyy[-2,:]
        detjac = phiinvxx*phiinvyy - phiinvxy*phiinvyx
        
        # pull back the gradient, it is multiplied by detjac because this 
        # operation is transport of measure
        interpolator.values = lambdaI
        lambdaIt = interpolator((evaly,evalx))*detjac
        
        # and image derivative, centered difference
        Itx = (np.roll(It[t],-1,axis=1) - np.roll(It[t],1,axis=1))/(2.0*dx)
        Ity = (np.roll(It[t],-1,axis=0) - np.roll(It[t],1,axis=0))/(2.0*dx)
        Itx[:,0] = Itx[:,1]
        Itx[:,-1] = Itx[:,-2]
        Ity[0,:] = Ity[1,:]
        Ity[-1,:] = Ity[-2,:]
        
        # the matching part of the gradient        
        gradx = -lambdaIt*Itx
        grady = -lambdaIt*Ity
        
        # and smooth it
        gradx = np.real(np.fft.ifft2(np.fft.fft2(gradx)*Khat))
        grady = np.real(np.fft.ifft2(np.fft.fft2(grady)*Khat))
        
        # and add the regularization term
        gradx += vtx[t]/sigmaR**2
        grady += vty[t]/sigmaR**2
                
        # gradient descent update
        vtx[t] -= epsilon*gradx
        vty[t] -= epsilon*grady
        