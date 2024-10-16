import numpy as np
from scipy.signal import hilbert
import numba as nb

def phases(t_series):
    """It computs the phases of the oscillators
    from the time series data of all the nodes.

    The input must be in the form of an (time x nodes) array
    
    The return data has the form (nodes x time)"""

    num_nodes = len(t_series[0])
    num_tpoints = len(t_series)
    analytic_phase = np.zeros(shape = (num_nodes, num_tpoints))
    for node in range(num_nodes):
        fase = np.array(t_series[:, node])
        analytic_signal = hilbert(fase)
        analytic_phase[node][:] = np.unwrap(np.angle(analytic_signal))


    #transpose of the t_series
    # t_series_t  = np.transpose(t_series)
    
    # analytic_signal = hilbert(t_series_t)
    # analytic_phase = np.unwrap(np.angle(analytic_signal))
    
    return analytic_phase

def Kuramoto(phases):
    """the parameter 'phases' has to be array like (nodes x time)"""
    kuramoto = abs(np.average(np.exp(1j*phases), axis=0))

    
    return kuramoto


def Cos_distance(phase):
    # Reshape array to match broadcasting dimensions
    reshaped_phase = phase[:, np.newaxis, :]
    
    # Compute cosine of the subtraction using broadcasting
    cos_dist = np.cos(reshaped_phase - phase)
    
    cos_dist = cos_dist.transpose(2, 0, 1) #first index is time, second and third nodes
    
    return cos_dist

def Leida(phase):
    
    cos_dist = Cos_distance(phase)
    
    eigen_values, eigen_vectors = np.linalg.eig(cos_dist)
    max_eigenval = np.argmax(eigen_values, axis = 1)
    

    v = np.arange(len(max_eigenval))

    max_eigenvector = eigen_vectors[v, :, max_eigenval]

    #result = np.matmul(max_eigenvector, max_eigenvector.T)

    return max_eigenvector
    

    