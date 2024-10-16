import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numba as nb
from numpy.linalg import slogdet

def FC_cal(t_series):

    """Calculates the Functional Connectivity of a time series

    The input must be in the form of an (time x nodes) array"""

    N = len(t_series[0]) #number of nodes
    #FC = np.zeros( shape = (N, N))  #funciontal connectivity

    print("FC is being calculated...")
    
    FC = np.corrcoef(t_series, rowvar = False)

    return FC



def FCD_cal(t_series, time_window_duration, time_window_overlap, tstep_of_data):

    """It splits the Time Series into windows and
    calculates the FC for a window of time.

    time_window_duration is the time length of the window.
    time_window_overlap is the lenght of overlapping between one temporal window and the next;
    tstep_of_data is the time lenght of time between data points"""

    print('The FCD is being calculated...')
    N = len(t_series[0]) #number of nodes
    n = len(t_series)    #number of time points
    num_windows = int((n*tstep_of_data -time_window_overlap)/(time_window_duration-time_window_overlap))
    print('Number of Windows:', num_windows)

    FC_temp = np.zeros(shape=(num_windows, N, N))   #array that contais all of the FCs
    FCD = np.zeros(shape=(num_windows, num_windows))
    FC_arrays = np.zeros(shape=(num_windows, int(N*(N-1)/2)))  #array that contains the upper diagonal
                                                          #of the FCs

    #initial time for the second window.
    # The 2 deviding is because there are half the points because of the 2 seconds data adquisition
    t_start = int((time_window_duration - time_window_overlap)/tstep_of_data)
    

    
    for i in range(num_windows):
        x = t_series[(i*t_start):(int(time_window_duration/tstep_of_data) + i*t_start), :]
        FC_temporal = np.corrcoef(x, rowvar = False)

        FC_arrays[i, :] = FC_temporal[np.triu_indices(N, k = 1)]

        

    FCD_new = np.corrcoef(FC_arrays)
    
    
    return FCD_new



def FC_temp_cal(t_series, time_window_duration, time_window_overlap, tstep_of_data):

    """It splits the Time Series into windows and
    calculates the FC for a window of time.

    time_window_duration is the time length of the window.
    time_window_overlap is the lenght of overlapping between one temporal window and the next;
    tstep_of_data is the time lenght of time between data points"""

    print('The FCD is being calculated...')
    N = len(t_series[0]) #number of nodes
    n = len(t_series)    #number of time points
    num_windows = int((n*tstep_of_data -time_window_overlap)/(time_window_duration-time_window_overlap))
    print('Number of Windows:', num_windows)

    FC_temporal = np.zeros(shape=(num_windows, N, N))   #array that contais all of the FCs
    


    #initial time for the second window.
    # The 2 deviding is because there are half the points because of the 2 seconds data adquisition
    t_start = int((time_window_duration - time_window_overlap)/tstep_of_data)
    

    
    for i in range(num_windows):
        x = t_series[(i*t_start):(int(time_window_duration/tstep_of_data) + i*t_start), :]
        FC = np.corrcoef(x, rowvar = False)
        FC_temporal[i] = FC


    
    return FC_temporal



def transfer_entropy_norm(X, NLags):
    

    N = X.shape[0]  #number of nodes
    MaxLag = np.max(NLags)
    Numdata = (X.shape[1] - MaxLag).astype(int)
    
    v = np.arange(MaxLag, 0,  -1)
    column_indices = np.tile(v, (Numdata, 1))
    j_index = np.arange(Numdata).reshape(-1, 1)
    column_indices = column_indices + j_index
    
    
    y_pre = X[:, column_indices]  
    
    GCsim = np.zeros(shape = (N, N))
    
    
    for i in range(N):
    
      y = np.squeeze(y_pre[i, :, 0:NLags[i,i]])
      y1 = np.squeeze(y_pre[i, :, 0])
       
      Hy1 = np.log(np.var(y1, ddof = 1, axis = 0))
      Iy = slogdet(np.cov(y, rowvar = False))[1] - slogdet(np.cov(y[:, 1:], rowvar = False))[1]
      
    
      nodelist = np.arange(N)
      nodelist = np.delete(nodelist, i)
      for k in nodelist:
        x = np.squeeze(y_pre[k, :, 1:NLags[i, k]])
        if len(x) == 1:
          x = x.T
    
        z = np.hstack((y, x))
        Iyxz = Iy - ( slogdet(np.cov(z, rowvar = False))[1] - slogdet(np.cov(z[:, 1:], rowvar = False))[1] )
        Iypast = Hy1 + slogdet(np.cov(z[:, 1:], rowvar = False))[1] - slogdet(np.cov(z, rowvar = False))[1]
    
        if Iyxz/Iypast < 0:
          Iyxz = 1e-15
          GCsim[i,k] = Iyxz
        else:
    
          GCsim[i,k] = Iyxz/Iypast
    
    
    return GCsim


def pair_granger_norm_6(X, NLags, length_windows, dt):

  N = X.shape[0]   #number of nodes 
  window_points = int(length_windows/dt)
  num_windows = int(X.shape[1]/window_points)

  GCsim_all = np.zeros(shape = (num_windows, N, N))
  for win in range(num_windows):

    X_new = X[:, win*window_points:(win+1)*window_points]
    MaxLag = np.max(NLags)
    Numdata = (X_new.shape[1] - MaxLag).astype(int)


    v = np.arange(MaxLag, 0,  -1)
    column_indices = np.tile(v, (Numdata, 1))
    j_index = np.arange(Numdata).reshape(-1, 1)
    column_indices = column_indices + j_index


    y_pre = X_new[:, column_indices]  
    GCsim = np.zeros(shape = (N, N))

    
    for i in range(N):

      y = np.squeeze(y_pre[i, :, 0:NLags[i,i]])
      y1 = np.squeeze(y_pre[i, :, 0])
      
      Hy1 = np.log(np.var(y1, ddof = 1, axis = 0))
      Iy = slogdet(np.cov(y, rowvar = False))[1] - slogdet(np.cov(y[:, 1:], rowvar = False))[1]
      


      nodelist = np.arange(N)
      nodelist = np.delete(nodelist, i)
      for k in nodelist:
        x = np.squeeze(y_pre[k, :, 1:NLags[i, k]])
        if len(x) == 1:
          x = x.T

        z = np.hstack((y, x))
        Iyxz = Iy - ( slogdet(np.cov(z, rowvar = False))[1] - slogdet(np.cov(z[:, 1:], rowvar = False))[1] )
        Iypast = Hy1 + slogdet(np.cov(z[:, 1:], rowvar = False))[1] - slogdet(np.cov(z, rowvar = False))[1]

        if Iyxz/Iypast < 0:
          Iyxz = 1e-15
          GCsim[i,k] = Iyxz
        else:

          GCsim[i,k] = Iyxz/Iypast
    
    GCsim_all[win] = GCsim

  return GCsim_all

def FCD_partitions(t_series, time_window_duration, time_window_overlap, tstep_of_data, partitions):

    """It splits the Time Series into windows and
    calculates the FC for a window of time.

    time_window_duration is the time length of the window.
    time_window_overlap is the lenght of overlapping between one temporal window and the next;
    tstep_of_data is the time lenght of time between data points"""

    print('The FCD is being calculated...')
    N = len(t_series[0]) #number of nodes
    n = len(t_series)    #number of time points
    num_windows = int((n*tstep_of_data -time_window_overlap)/(time_window_duration-time_window_overlap))
    num_partitions = partitions.shape[1]
    print('Number of Windows:', num_windows)

    FC_temp = np.zeros(shape=(num_windows, N, N))   #array that contais all of the FCs
    FCD = np.zeros(shape=(num_partitions, num_windows, num_windows))

    #initial time for the second window.
    # The 2 deviding is because there are half the points because of the 2 seconds data adquisition
    t_start = int((time_window_duration - time_window_overlap)/tstep_of_data)

    for net in range(num_partitions):
        partition_indx = np.nonzero(partitions[:, net])
        
        series_partition = t_series[:, partition_indx].squeeze()
        num_partition_nodes = series_partition.shape[1]
        
        FC_arrays = np.zeros(shape=(num_windows, int(num_partition_nodes*(num_partition_nodes-1)/2)))
        for i in range(num_windows):
            x = series_partition[(i*t_start):(int(time_window_duration/tstep_of_data) + i*t_start), :]
            FC_temporal = np.corrcoef(x, rowvar = False)
    
            FC_arrays[i, :] = FC_temporal[np.triu_indices(num_partition_nodes, k = 1)]
    
        FCD[net] = np.corrcoef(FC_arrays)
            

    
    return FCD
