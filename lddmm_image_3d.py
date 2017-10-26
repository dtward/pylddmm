# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:22:04 2017

@author: dtward

3d lddmm

TO DO: template estimation

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp


def gradient(I,dx,dy,dz):
    '''
    Takes centered difference in middle, forward or backward difference at edges
    last index is x, second last is y, third last is z
    Should be able to work with multi channel images based on indexing with ellipses
    '''
    Ix = (np.roll(I,-1,axis=-1) - np.roll(I,1,axis=-1))/(2.0*dx)
    Iy = (np.roll(I,-1,axis=-2) - np.roll(I,1,axis=-2))/(2.0*dy)
    Iz = (np.roll(I,-1,axis=-3) - np.roll(I,1,axis=-3))/(2.0*dz)
    
    # basic boundary conditions
    '''
    Ix[:,:,0] = Ix[:,:,1]
    Ix[:,:,-1] = Ix[:,:,-2]
    Iy[:,0,:] = Iy[:,1,:]
    Iy[:,-1,:] = Iy[:,-2,:]
    Iz[0,:,:] = Iz[1,:,:]
    Iz[-1,:,:] = Iz[-2,:,:]
    '''
    # instead, do like matlab, no centered difference at edges
    Ix[...,0] = (I[...,1] - I[...,0])/dx
    Ix[...,-1] = (I[...,-1] - I[...,-2])/dx
    Iy[...,0,:] = (I[...,1,:] - I[...,0,:])/dy
    Iy[...,-1,:] = (I[...,-1,:] - I[...,-2,:])/dy
    Iz[0,:,:] = (I[1,:,:] - I[0,:,:])/dz
    Iz[-1,:,:] = (I[-1,:,:] - I[-2,:,:])/dz
    
    # return the gradient as a tuple
    return Ix,Iy,Iz
    
def determinant_of_jacobian(phix,phiy,phiz,dx,dy,dz):
    phixx,phixy,phixz = gradient(phix,dx,dy,dz)
    phiyx,phiyy,phiyz = gradient(phiy,dx,dy,dz)
    phizx,phizy,phizz = gradient(phiz,dx,dy,dz)
    
    return phixx*(phiyy*phizz - phiyz*phizy) \
        - phixy*(phiyx*phizz - phiyz*phizx) \
        + phixz*(phiyx*phizy - phiyy*phizx)
    
    
def lddmm_image_3d(xA,yA,zA,IA,xT,yT,zT,IT,sigmaI=0.1,sigmaR=10.0,alpha=10.0,nT=5,niter=100,epsilon=0.1,nshow=10,nprint=1,vtx=None,vty=None,vtz=None):
    # set up figure
    if nshow: fig = plt.figure()    
    
    # image domain info
    dxT = xT[1]-xT[0]
    dyT = yT[1]-yT[0]
    dzT = zT[1]-zT[0]
    nxT = len(xT)
    nyT = len(yT)
    nzT = len(zT)        
        
    xxT = xT[None,None,:] + yT[None,:,None]*0 + zT[:,None,None]*0
    yyT = xT[None,None,:]*0 + yT[None,:,None] + zT[:,None,None]*0
    zzT = xT[None,None,:]*0 + yT[None,:,None]*0 + zT[:,None,None]
    
    # we also need frequency domain
    # and lowpass and highpass operator
    fx = np.arange(nxT)/dxT/nxT # note divide by dx first because it is float
    fy = np.arange(nyT)/dyT/nyT
    fz = np.arange(nzT)/dzT/nzT
    fxx = fx[None,None,:] + fy[None,:,None]*0 + fz[:,None,None]*0
    fyy = fx[None,None,:]*0 + fy[None,:,None] + fz[:,None,None]*0
    fzz = fx[None,None,:]*0 + fy[None,:,None]*0 + fz[:,None,None]
    
    # identity minus laplacian in Fourier domain
    # AI[i,j] = I[i,j] - alpha^2( (I[i+1,j] - 2I[i,j] + I[i-1,j])/dx^2 + (I[i,j+1] - 2I[i,j] + I[i,j-1])/dy^2  )
    Lhat = 1.0 - alpha**2*((-2.0 + 2.0*np.cos(2*np.pi*dxT*fxx))/dxT**2 
        + (-2.0 + 2.0*np.cos(2*np.pi*dyT*fyy))/dyT**2 
        + (-2.0 + 2.0*np.cos(2*np.pi*dzT*fzz))/dzT**2   )
    Ahat = Lhat**2
    Khat = 1.0/Ahat
       
    # now initialize our transformation
    # note that the transformation will match the sampling of J
    # these are the points we will need to evaluate our deformed image at in order to calculate error
    dt = 1.0/nT
    # I need to save v as a function of t to optimize over
    if vtx is None:
        vtx = np.zeros((nT,nzT,nyT,nxT))
    if vty is None:
        vty = np.zeros((nT,nzT,nyT,nxT))
    if vtz is None:
        vtz = np.zeros((nT,nzT,nyT,nxT))
    
    
    # interpolation will be used for deforming an image, 
    # and constructing diffeomorphisms
    # note when deforming two things with the same transformation, we want to 
    # do both at the same time
    interpolatorA = interp.RegularGridInterpolator((zA,yA,xA),IA,method='linear',bounds_error=False,fill_value=0.0)
    interpolatorT = interp.RegularGridInterpolator((zT,yT,xT),IA,method='linear',bounds_error=False,fill_value=0.0)
    
    
    # I need to save I as a function of t because I will need its gradient
    # note that in certain sparse sampling schemes it will be better to calculate 
    # the gradient first, and then deform it (push forward of vector field)
    It = np.zeros((nT+1,nzT,nyT,nxT))
    It[0] = interpolatorA((zzT,yyT,xxT))
    
    # start a gradient descent optimzation loop
    Er = []
    Em = []
    E = []
    for iteration in xrange(niter):    
        # flow forward
        reg_cost = 0.0 # initialize flow energy    
        # initialize diffeomorphisms, we do not need to save them as 
        # a function of t
        phi0tinvx = np.array(xxT) 
        phi0tinvy = np.array(yyT)
        phi0tinvz = np.array(zzT)
        for t in xrange(nT):
            # update the diffeomorphism
            # here we say phi(x) - identity(x) should have 0 boundary conditions
            # we evaluate it at x - v(x)*dt, then we add back identity(x - v(x)*dt)
            evalx = xxT-vtx[t]*dt
            evaly = yyT-vty[t]*dt
            evalz = zzT-vtz[t]*dt
            values = np.empty((nzT,nyT,nxT,3))# to evaluate both at once, I need this order
            values[:,:,:,0] = phi0tinvx-xxT
            values[:,:,:,1] = phi0tinvy-yyT
            values[:,:,:,2] = phi0tinvz-zzT
            interpolatorT.values = values
            out = interpolatorT((evalz,evaly,evalx))
            phi0tinvx = out[:,:,:,0]+xxT-vtx[t]*dt
            phi0tinvy = out[:,:,:,1]+yyT-vty[t]*dt
            phi0tinvz = out[:,:,:,2]+zzT-vtz[t]*dt
            # okay now deform the image
            interpolatorA.values = IA
            It[t+1] = interpolatorA((phi0tinvz,phi0tinvy,phi0tinvx))
            
            # energy of the flow
            Avx = np.real(np.fft.ifftn(np.fft.fftn(vtx[t])*Ahat))
            Avy = np.real(np.fft.ifftn(np.fft.fftn(vty[t])*Ahat))
            Avz = np.real(np.fft.ifftn(np.fft.fftn(vtz[t])*Ahat))
            reg_cost += 0.5*np.sum(vtx[t]*Avx + vty[t]*Avy + vtz[t]*Avz)/sigmaR**2*dxT*dyT*dzT*dt
            
        # now we get the cost
        Er.append(reg_cost)        
        match_cost = 0.5*np.sum((It[-1] - IT)**2)/sigmaI**2*dxT*dyT*dzT
        Em.append(match_cost)
        energy = reg_cost + match_cost
        E.append(energy)
        if nprint and not iteration%nprint:
            print('iteration {}, energy {} (reg {} + match {}))'.format(iteration, energy, reg_cost, match_cost))
        
        # initialize the gradient of the matching term
        lambdaI = (It[-1] - IT)/sigmaI**2
        
        # make a nice picture
        if nshow and not iteration%nshow:
            plt.clf()
            extent = (xT[0],xT[-1],yT[0],yT[-1])
            ax = fig.add_subplot(231)
            ax.imshow(It[t+1][nzT/2],extent=extent,interpolation='none',cmap='gray')
            ax = fig.add_subplot(234)
            h = ax.imshow(lambdaI[nzT/2],extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)
            
            extent = (xT[0],xT[-1],zT[0],zT[-1])
            ax = fig.add_subplot(232)
            ax.imshow(It[t+1][:,nyT/2,:],extent=extent,interpolation='none',cmap='gray')
            ax = fig.add_subplot(235)
            h = ax.imshow(lambdaI[:,nyT/2,:],extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)     
            
            extent = (yT[0],yT[-1],zT[0],zT[-1])
            ax = fig.add_subplot(233)
            ax.imshow(It[t+1][:,:,nxT/2],extent=extent,interpolation='none',cmap='gray')
            ax = fig.add_subplot(236)
            h = ax.imshow(lambdaI[:,:,nxT/2],extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)     

            plt.pause(1.0e-10) # draw it now
        
        # now we will backpropogate the gradient backward in time
        phi1tinvx = np.array(xxT)
        phi1tinvy = np.array(yyT)
        phi1tinvz = np.array(zzT)
        for t in range(nT-1,-1,-1):
            # construct the diffeomorphism
            evalx = xxT + vtx[t]*dt
            evaly = yyT + vty[t]*dt
            evalz = zzT + vtz[t]*dt
            values = np.empty((nzT,nyT,nxT,3))# to evaluate both at once, I need this order
            values[:,:,:,0] = phi1tinvx-xxT
            values[:,:,:,1] = phi1tinvy-yyT
            values[:,:,:,2] = phi1tinvz-zzT
            interpolatorT.values = values
            out = interpolatorT((evalz,evaly,evalx))
            phi1tinvx = out[:,:,:,0]+xxT+vtx[t]*dt
            phi1tinvy = out[:,:,:,1]+yyT+vty[t]*dt
            phi1tinvz = out[:,:,:,2]+zzT+vtz[t]*dt
            
            # we need determinant of jacobian, use centered difference
            '''
            phi1tinvxx = (np.roll(phi1tinvx,-1,axis=2) - np.roll(phi1tinvx,1,axis=2))/(2.0*dxT)
            phi1tinvxy = (np.roll(phi1tinvx,-1,axis=1) - np.roll(phi1tinvx,1,axis=1))/(2.0*dyT)
            phi1tinvxz = (np.roll(phi1tinvx,-1,axis=0) - np.roll(phi1tinvx,1,axis=0))/(2.0*dzT)
            
            phi1tinvyx = (np.roll(phi1tinvy,-1,axis=2) - np.roll(phi1tinvy,1,axis=2))/(2.0*dxT)
            phi1tinvyy = (np.roll(phi1tinvy,-1,axis=1) - np.roll(phi1tinvy,1,axis=1))/(2.0*dyT)
            phi1tinvyz = (np.roll(phi1tinvy,-1,axis=0) - np.roll(phi1tinvy,1,axis=0))/(2.0*dzT)
            
            phi1tinvzx = (np.roll(phi1tinvz,-1,axis=2) - np.roll(phi1tinvz,1,axis=2))/(2.0*dxT)
            phi1tinvzy = (np.roll(phi1tinvz,-1,axis=1) - np.roll(phi1tinvz,1,axis=1))/(2.0*dyT)
            phi1tinvzz = (np.roll(phi1tinvz,-1,axis=0) - np.roll(phi1tinvz,1,axis=0))/(2.0*dzT)
            
            # careful boundary conditions
            phi1tinvxx[:,:,0] = phi1tinvxx[:,:,1]
            phi1tinvxx[:,:,-1] = phi1tinvxx[:,:,-2]
            phi1tinvxy[:,0,:] = phi1tinvxy[:,1,:]
            phi1tinvxy[:,-1,:] = phi1tinvxy[:,-2,:]
            phi1tinvxz[0,:,:] = phi1tinvxz[1,:,:]
            phi1tinvxz[-1,:,:] = phi1tinvxz[-2,:,:]
            
            phi1tinvyx[:,:,0] = phi1tinvyx[:,:,1]
            phi1tinvyx[:,:,-1] = phi1tinvyx[:,:,-2]
            phi1tinvyy[:,0,:] = phi1tinvyy[:,1,:]
            phi1tinvyy[:,-1,:] = phi1tinvyy[:,-2,:]
            phi1tinvyz[0,:,:] = phi1tinvyz[1,:,:]
            phi1tinvyz[-1,:,:] = phi1tinvyz[-2,:,:]
            
            phi1tinvzx[:,:,0] = phi1tinvzx[:,:,1]
            phi1tinvzx[:,:,-1] = phi1tinvzx[:,:,-2]
            phi1tinvzy[:,0,:] = phi1tinvzy[:,1,:]
            phi1tinvzy[:,-1,:] = phi1tinvzy[:,-2,:]
            phi1tinvzz[0,:,:] = phi1tinvzz[1,:,:]
            phi1tinvzz[-1,:,:] = phi1tinvzz[-2,:,:]
            '''
            '''
            phi1tinvxx,phi1tinvxy,phi1tinvxz = gradient(phi1tinvx,dxT,dyT,dzT)
            phi1tinvyx,phi1tinvyy,phi1tinvyz = gradient(phi1tinvy,dxT,dyT,dzT)
            phi1tinvzx,phi1tinvzy,phi1tinvzz = gradient(phi1tinvz,dxT,dyT,dzT)
            
            detjac = phi1tinvxx*(phi1tinvyy*phi1tinvzz - phi1tinvyz*phi1tinvzy)\
                -phi1tinvxy*(phi1tinvyx*phi1tinvzz - phi1tinvyz*phi1tinvzx)\
                + phi1tinvxz*(phi1tinvyx*phi1tinvzy - phi1tinvyy*phi1tinvzx)
            '''
            detjac = determinant_of_jacobian(phi1tinvx,phi1tinvy,phi1tinvz,dxT,dyT,dzT)
            # pull back the gradient, it is multiplied by detjac because this 
            # operation is transport of measure
            interpolatorT.values = lambdaI
            lambdaIt = interpolatorT((phi1tinvz,phi1tinvy,phi1tinvx))*detjac
            
            # and image derivative, centered difference
            '''
            Itx = (np.roll(It[t],-1,axis=2) - np.roll(It[t],1,axis=2))/(2.0*dxT)
            Ity = (np.roll(It[t],-1,axis=1) - np.roll(It[t],1,axis=1))/(2.0*dyT)
            Itz = (np.roll(It[t],-1,axis=0) - np.roll(It[t],1,axis=0))/(2.0*dzT)
            
            
            Itx[:,:,0] = Itx[:,:,1]
            Itx[:,:,-1] = Itx[:,:,-2]
            Ity[:,0,:] = Ity[:,1,:]
            Ity[:,-1,:] = Ity[:,-2,:]
            Itz[0,:,:] = Itz[1,:,:]
            Itz[-1,:,:] = Itz[-2,:,:]
            '''
            # replace with my new gradient function
            Itx,Ity,Itz = gradient(It[t],dxT,dyT,dzT)
            
            # the matching part of the gradient        
            gradx = -lambdaIt*Itx
            grady = -lambdaIt*Ity
            gradz = -lambdaIt*Itz
            
            # and smooth it
            gradx = np.real(np.fft.ifftn(np.fft.fftn(gradx)*Khat))
            grady = np.real(np.fft.ifftn(np.fft.fftn(grady)*Khat))
            gradz = np.real(np.fft.ifftn(np.fft.fftn(gradz)*Khat))
            
            # and add the regularization term
            gradx += vtx[t]/sigmaR**2
            grady += vty[t]/sigmaR**2
            gradz += vtz[t]/sigmaR**2
                    
            # gradient descent update
            vtx[t] -= epsilon*gradx
            vty[t] -= epsilon*grady
            vtz[t] -= epsilon*gradz
    # note that lambdaI0 is the gradient of the cost with respect to the image
    # this is useful for template estimation

    return {'vtx':vtx,'vty':vty,'vtz':vtz,
            'It':It,
            'phi01invx':phi0tinvx,'phi01invy':phi0tinvy,'phi01invz':phi0tinvz,
            'phi10invx':phi1tinvx,'phi10invy':phi1tinvy,'phi10invz':phi1tinvz,
            'lambdaI0':lambdaIt,'detjacphi10inv':detjac,
            'Er':Er,'Em':Em,'E':E}
            

def lddmm_image_3d_template(x,y,z,I,sigmaI=0.1,sigmaR=10.0,alpha=10.0,nT=5,niter=100,epsilon=0.1,nshow=10,IA=None,niterA=100,epsilonA=0.1,vtx=None,vty=None,vtz=None,tvreg = 1.0,tvepsilon=0.01):
    ''' 
    for inputs assume for now they are all the same size
    I will be a LIST of images (or an equivalent array whose first index gives an image)
    '''
    if nshow is not None:
        fig = plt.figure()
    #figtest = plt.figure()
    #axtest = figtest.add_subplot(111)
    N = len(I)
    if IA is None:
        IA = np.zeros_like(I[0]) 
        for i in range(N):
            IA += I[i]
        IA /= float(N)
    #IA *= 0
    
    nx = IA.shape[-1]
    ny = IA.shape[-2]
    nz = IA.shape[-3]
    
    if vtx is None:
        vtx = np.zeros((N,nT,nz,ny,nx))
    if vty is None:
        vty = np.zeros((N,nT,nz,ny,nx))
    if vtz is None:
        vtz = np.zeros((N,nT,nz,ny,nx))
        
    Em = []
    Er = []
    E = []
    for iteration in range(niterA):
        IAgrad = np.zeros_like(IA)
        matching_cost = 0.0
        reg_cost = 0.0        
        for i in range(N):
            output = lddmm_image_3d(x,y,z,IA,x,y,z,I[i],sigmaI=sigmaI,sigmaR=sigmaR,alpha=alpha,nT=nT,niter=niter,epsilon=epsilon,nshow=0,vtx=vtx[i],vty=vty[i],vtz=vtz[i],nprint=0)
            vtx[i] = output['vtx']
            vty[i] = output['vty']
            vtz[i] = output['vtz']
            IAgrad = IAgrad + output['lambdaI0']

            matching_cost += output['Em'][-1]
            reg_cost += output['Er'][-1]
            
            
            
            
            # test
            #figtest.clf()
            #axtest = figtest.add_subplot(111)
            #axtest.imshow(output['detjacphi10inv'][nz/2],cmap='gray',interpolation='none')
            #plt.pause(0.01)
        Em.append(matching_cost)
        Er.append(reg_cost)
        energy = matching_cost + reg_cost
        E.append(energy)
        print('iteration {}, energy {} (reg {} + match {}))'.format(iteration, energy, reg_cost, matching_cost))
        
        # regularization?
        # I probably want tv regularization
        # if I'm not doing regularization, this variational approach I think is dumb

        # hack, problem on edges of gradient, I don't konw why
        IAgrad[0,:,:] = IAgrad[-1,:,:] = IAgrad[:,0,:] = IAgrad[:,-1,:] = IAgrad[:,:,0] = IAgrad[:,:,-1] = 0
        
        IA = IA - epsilonA*IAgrad
        if nshow and not iteration%nshow:
            fig.clf()
            clim = (0,1)
            clim = None
            extent = (x[0],x[-1],y[0],y[-1])            
            ax = fig.add_subplot(231)
            h = ax.imshow(IA[nz/2],extent=extent,interpolation='none',cmap='gray',clim=clim)
            plt.colorbar(mappable=h)
            
            extent = (x[0],x[-1],z[0],z[-1])
            ax = fig.add_subplot(232)
            h = ax.imshow(IA[:,ny/2,:],extent=extent,interpolation='none',cmap='gray',clim=clim)
            plt.colorbar(mappable=h)     
            
            extent = (y[0],y[-1],z[0],z[-1])
            ax = fig.add_subplot(233)
            h = ax.imshow(IA[:,:,nx/2],extent=extent,interpolation='none',cmap='gray',clim=clim)
            plt.colorbar(mappable=h)     
            
            vis = 'lambdaI0'
            vis = 'detjacphi10inv'
            vis = 'grad'

            output['grad'] = IAgrad
            
            extent = (x[0],x[-1],y[0],y[-1])       
            ax = fig.add_subplot(234)
            h = ax.imshow(output[vis][nz/2],extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)
            
            extent = (x[0],x[-1],z[0],z[-1])
            ax = fig.add_subplot(235)
            h = ax.imshow(output[vis][:,ny/2,:],extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)     
            
            extent = (y[0],y[-1],z[0],z[-1])
            ax = fig.add_subplot(236)
            h = ax.imshow(output[vis][:,:,nx/2],extent=extent,interpolation='none',cmap='gray')
            plt.colorbar(mappable=h)     

            fig.savefig('template_iteration_{:04d}.png'.format(iteration))
            plt.pause(1.0e-10) # draw it now
            