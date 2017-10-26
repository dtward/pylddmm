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


def lddmm_image_2d(xA,yA,IA,xT,yT,IT,sigmaI=0.1,sigmaR=10.0,alpha=10.0,nT=5,niter=100,epsilon=0.1,nshow=10):
    # image domain
    # in general this will be read from an input file
    # in general
    if nshow: fig = plt.figure()    
    
    dxT = xT[1]-xT[0]
    dyT = yT[1]-yT[0]
    nxT = len(xT)
    nyT = len(yT)
        
    
    xxA,yyA = np.meshgrid(xA,yA)
    xxT,yyT = np.meshgrid(xT,yT)
    
    # we also need frequency domain
    # and lowpass and highpass operator
    fx = np.arange(nxT)/dxT/nxT # note divide by dx first because it is float
    fy = np.arange(nyT)/dyT/nyT
    fxx,fyy = np.meshgrid(fx,fy)
    # AI[i,j] = I[i,j] - alpha^2( (I[i+1,j] - 2I[i,j] + I[i-1,j])/dx^2 + (I[i,j+1] - 2I[i,j] + I[i,j-1])/dy^2  )
    Lhat = 1.0 - alpha**2*((-2.0 + 2.0*np.cos(2*np.pi*dxT*fxx))/dxT**2 + (-2.0 + 2.0*np.cos(2*np.pi*dyT*fyy))/dyT**2   )
    Ahat = Lhat**2
    Khat = 1.0/Ahat
       
    # now initialize our transformation
    # note that the transformation will match the sampling of J
    # these are the points we will need to evaluate our deformed image at in order to calculate error
    dt = 1.0/nT
    # I need to save v as a function of t to optimize over
    vtx = np.zeros((nT,nyT,nxT))
    vty = np.zeros((nT,nyT,nxT))
    
    
    # interpolation will be used for deforming an image, 
    # and constructing diffeomorphisms
    # note when deforming two things with the same transformation, we want to do 
    # both at the same time
    interpolatorA = interp.RegularGridInterpolator((yA,xA),IA,method='linear',bounds_error=False,fill_value=0.0)
    interpolatorT = interp.RegularGridInterpolator((yT,xT),IA,method='linear',bounds_error=False,fill_value=0.0)
    
    
    # I need to save I as a function of t because I will need its gradient
    # note that in certain sparse sampling schemes it will be better to calculate 
    # the gradient first, and then deform it (push forward of vector field)
    It = np.zeros((nT+1,nyT,nxT))
    It[0] = interpolatorA((yyT,xxT))
    
    
    
    
    # start a gradient descent optimzation loop
    for iteration in xrange(niter):    
        # flow forward
        reg_cost = 0.0 # initialize flow energy    
        phi0tinvx = np.array(xxT) # initialize diffeomorphisms
        phi0tinvy = np.array(yyT)
        for t in xrange(nT):
            # update the diffeomorphism
            # here we say phi(x) - identity(x) should have 0 boundary conditions
            # we evaluate it at x - v(x)*dt, then we add back identity(x - v(x)*dt)
            evalx = xxT-vtx[t]*dt
            evaly = yyT-vty[t]*dt
            values = np.empty((nyT,nxT,2))# to evaluate both at once, I need this order
            values[:,:,0]  = phi0tinvx-xxT
            values[:,:,1] = phi0tinvy-yyT      
            interpolatorT.values = values
            out = interpolatorT((evaly,evalx))
            phi0tinvx = out[:,:,0]+xxT-vtx[t]*dt
            phi0tinvy = out[:,:,1]+yyT-vty[t]*dt
            # okay now deform the image
            interpolatorA.values = IA
            It[t+1] = interpolatorA((phi0tinvy,phi0tinvx))
            
            # energy of the flow
            Avx = np.real(np.fft.ifft2(np.fft.fft2(vtx[t])*Ahat))
            Avy = np.real(np.fft.ifft2(np.fft.fft2(vty[t])*Ahat))
            reg_cost += 0.5*np.sum(vtx[t]*Avx + vty[t]*Avy)/sigmaR**2*dxT*dyT*dt
            
        # now we get the cost
        match_cost = 0.5*np.sum((It[-1] - IT)**2)/sigmaI**2*dxT*dyT
        energy = reg_cost + match_cost
        print('iteration {}, energy {} (reg {} + match {}))'.format(iteration, energy, reg_cost, match_cost))
        
        # initialize the gradient of the matching term
        lambdaI = (It[-1] - IT)/sigmaI**2
        
        # make a nice picture
        if nshow and not iteration%nshow:
            plt.clf()
            extent = (xT[0],xT[-1],yT[0],yT[-1])
            ax0 = fig.add_subplot(121)
            ax0.imshow(It[t+1],extent=extent,interpolation='none',cmap='gray')
            ax1 = fig.add_subplot(122)
            h = ax1.imshow(lambdaI,extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)
            plt.pause(1.0e-10) # draw it now
        
        # now we will backpropogate the gradient backward in time
        phi1tinvx = np.array(xxT)
        phi1tinvy = np.array(yyT)
        for t in range(nT-1,-1,-1):
            # construct the diffeomorphism
            evalx = xxT + vtx[t]*dt
            evaly = yyT + vty[t]*dt
            values = np.empty((nyT,nxT,2))# to evaluate both at once, I need this order
            values[:,:,0] = phi1tinvx-xxT
            values[:,:,1] = phi1tinvy-yyT
            interpolatorT.values = values
            out = interpolatorT((evaly,evalx))
            phi1tinvx = out[:,:,0]+xxT+vtx[t]*dt
            phi1tinvy = out[:,:,1]+yyT+vty[t]*dt
            
            # we need determinant of jacobian, use centered difference
            phi1tinvxx = (np.roll(phi1tinvx,-1,axis=1) - np.roll(phi1tinvx,1,axis=1))/(2.0*dxT)
            phi1tinvxy = (np.roll(phi1tinvx,-1,axis=0) - np.roll(phi1tinvx,1,axis=0))/(2.0*dyT)
            phi1tinvyx = (np.roll(phi1tinvy,-1,axis=1) - np.roll(phi1tinvy,1,axis=1))/(2.0*dxT)
            phi1tinvyy = (np.roll(phi1tinvy,-1,axis=0) - np.roll(phi1tinvy,1,axis=0))/(2.0*dyT)
            # careful boundary conditions
            phi1tinvxx[:,0] = phi1tinvxx[:,1]
            phi1tinvxx[:,-1] = phi1tinvxx[:,-2]
            phi1tinvxy[0,:] = phi1tinvxy[1,:]
            phi1tinvxy[-1,:] = phi1tinvxy[-2,:]
            phi1tinvyx[:,0] = phi1tinvyx[:,1]
            phi1tinvyx[:,-1] = phi1tinvyx[:,-2]
            phi1tinvyy[0,:] = phi1tinvyy[1,:]
            phi1tinvyy[-1,:] = phi1tinvyy[-2,:]
            detjac = phi1tinvxx*phi1tinvyy - phi1tinvxy*phi1tinvyx
            
            # pull back the gradient, it is multiplied by detjac because this 
            # operation is transport of measure
            interpolatorT.values = lambdaI
            lambdaIt = interpolatorT((phi1tinvy,phi1tinvx))*detjac
            
            # and image derivative, centered difference
            Itx = (np.roll(It[t],-1,axis=1) - np.roll(It[t],1,axis=1))/(2.0*dxT)
            Ity = (np.roll(It[t],-1,axis=0) - np.roll(It[t],1,axis=0))/(2.0*dyT)
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
    return vtx,vty,It,phi0tinvx,phi0tinvy,phi1tinvx,phi1tinvy