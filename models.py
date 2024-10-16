import numpy as np
from time import process_time
import time
import numba as nb
import dask.array as da





@nb.jit(nopython=True)
def hopf_brain_faster(X, t, M,  G, frequencies, constant, a, F= None, kinetic = None):
    if F is None:
        numNodes = len(M)
        omega = frequencies
        
        z = X[0]
    
        
    
    
        zz = z*np.conjugate(z)
        parentesis = (a + 1j*omega - zz)
    
        
        #without delay
        fz = z*parentesis + G*(np.dot(M,z)  -constant*z) 
    
    else:
        numNodes = len(M)
        omega = frequencies
        
        z = X[0]
    
        
    
    
        zz = z*np.conjugate(z)
        parentesis = (a + 1j*omega - zz)
    
        
        #without delay
        fz = z*parentesis + G*(np.dot(M,z)  -constant*z) + F

    
    return fz


def hopf_brain(X, t, M,  G, frequencies, constant, a, F= None, kinetic = None):
    if F is None:
        numNodes = len(M)
        omega = frequencies
        
        z = X[0]
    
        
    
    
        zz = z*np.conjugate(z)
        parentesis = (a + 1j*omega - zz)
    
        
        #without delay
        fz = z*parentesis + G*(np.dot(M,z)  -constant*z) 
    
    else:
        numNodes = len(M)
        omega = frequencies
        
        z = X[0]
    
        
    
    
        zz = z*np.conjugate(z)
        parentesis = (a + 1j*omega - zz)
    
        
        #without delay
        fz = z*parentesis + G*(np.dot(M,z)  -constant*z) + F

    
    return fz




def Hopf_extended(X, t,  M, wx_00, wx_10, wx_01, wx_20, wx_11, wx_02, wx_30, wx_21, wx_12, wx_03, wx_40, wx_31,  wx_22, wx_13, wx_04, \
        
         wx_50, wx_41, wx_32, wx_23, wx_14, wx_05, wy_00, wy_10, wy_01, wy_20, wy_11, wy_02, wy_30, wy_21, wy_12, wy_03, wy_40, wy_31,  wy_22, wy_13, wy_04, \
        
         wy_50, wy_41, wy_32, wy_23, wy_14, wy_05, G, K):
    
    numNodes = len(M)
    F = np.zeros( shape = (2, numNodes))
    x, y = X[0,:], X[1,:]
    
    fx = wx_00 + wx_10*x + wx_01*y + wx_20*x**2 +  wx_11*x*y + wx_02*y**2  +  \
        + wx_30*x**3 +  wx_21*(x**2)*y + wx_12*x*y**2 + wx_03*y**3 + \
               + wx_40*x**4 + wx_31*(x**3)*y + wx_22*(x**2)*(y**2) + wx_13*x*(y**3)  + wx_04*y**4 + \
         + wx_50*x**5+ wx_41*(x**4)*y + wx_32*(x**3)*(y**2) + wx_23*(x**2)*(y**3) + wx_14*x*(y**4) +  wx_05*y**5 + \
          +   G*(np.dot(M, x)  - K*x) 
   
    
    fy = wy_00 + wy_10*x + wy_01*y + wy_20*x**2 +  wy_11*x*y + wy_02*y**2 +  \
         + wy_30*x**3 +  wy_21*(x**2)*y + wy_12*x*y**2 + wy_03*y**3 + \
               + wy_40*x**4 + wy_31*(x**3)*y + wy_22*(x**2)*(y**2) + wy_13*x*(y**3)  + wy_04*y**4 + \
        + wy_50*x**5+ wy_41*(x**4)*y + wy_32*(x**3)*(y**2) + wy_23*(x**2)*(y**3) + wy_14*x*(y**4) +  wy_05*y**5 + \
               + G*(np.dot(M, y)  - K*y) 
    
    F[0,:], F[1,:] = fx, fy
    
    return F

def rossler_brain(R, t,  M, G, frequencies, constant, a, b, c):
    
    F = np.zeros( shape = (3, len(M)))
    gamma = 0.3
    numNodes = len(M)
    omega = frequencies
    
    x, y, z = R[0,:], R[1,:], R[2,:]
    
 
    xz = x*z
    
    fx = -omega*y - z*gamma +  gamma*(G*(np.dot(M, x)  - constant*x))
    fy = omega*x + a*y*gamma 
    fz = gamma*(b + xz - z*c)
    
    F[0,:], F[1, :], F[2,:] = fx, fy, fz
        
    
    return F