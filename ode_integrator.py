import numpy as np
import numba as nb


def integration_ode_2(function, initial_conditions, tmin = 0, tmax = 1000, \
                      tstep = 0.1,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
    """ 'function' is the function to be integrated.
    
    initial_condition must be a matrix with shape = (number of variables, number of nodes) 
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the function other than time should be passed.

    It returns 3 matrices 'xpoints', 'ypoints' and 'zpoints' whose columns are the 
    
    different nodes and the rows denote time
 
    """

    num_steps = int((tmax-tmin)/tstep)
    num_nodes = len(initial_conditions[0])


    tpoints, xpoints, ypoints, zpoints  = np.arange(tmin, tmax, tstep), \
                                          np.zeros( shape = (num_steps, num_nodes)), \
                                          np.zeros( shape = (num_steps, num_nodes)), \
                                          np.zeros( shape = (num_steps, num_nodes))
    R = initial_conditions

    if method == 'runge_kutta2':
      print('Method of integration: ', method)
      for t in range(num_steps):
        xpoints[t][::] = R[0][::]
        ypoints[t][::] = R[1][::]
        zpoints[t][::] = R[2][::]
        k1 = tstep*function(R, t, **kwargs)
        k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
        R += k2
   
    elif method == 'euler':
      print('Method of integration: ', method)
      for t in range(num_steps):
        xpoints[t][::] = R[0][::]
        ypoints[t][::] = R[1][::]
        zpoints[t][::] = R[2][::]
        R += tstep*function(R, t, **kwargs)
        
        
    elif method == 'euler_maruyama':
      print('Method of integration: ', method)
      for t in range(num_steps):
        dW = np.random.randn(1, num_nodes)
        xpoints[t][::] = R[0][::]
        ypoints[t][::] = R[1][::]
        zpoints[t][::] = R[2][::]
        R += tstep*function(R, t, **kwargs) + noise_amp*dW
      
    return xpoints, ypoints, zpoints

def integration_ode(function, initial_conditions, tmin, tmax, 
                    tstep,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
        
    """ 'function' is the function to be integrated.
    
    initial_condition must be an array of len(number of variables of the system)
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the system of equations (other than time) should be passed.

    It returns a matrix of the form X = (time, variable)
 
    """
    num_steps = int((tmax-tmin)/tstep)
    num_variables = len(initial_conditions)

    tpoints, X  = np.arange(tmin, tmax, tstep), np.zeros( shape = (num_steps, num_variables))

    R = initial_conditions

    if method == 'runge_kutta2':
      print('Method of integration: ', method)
      for t in range(num_steps):
          X[t][::] = R
          k1 = tstep*function(R, t, **kwargs)
          k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
          R += k2
    
    elif method == 'euler':
      print('Method of integration: ', method)
      for t in range(num_steps):
        X[t][::] = R
        R += tstep*function(R, t, **kwargs)
        


    elif method == 'euler_maruyama':
      print('Method of integration: ', method)
      for t in range(num_steps):
        dW = np.random.randn(1, num_nodes)
        X[t][::] = R
        R += tstep*function(R, t, **kwargs) + noise_amp*dW
      
    return X

def integration_ode_3(function, initial_conditions, tmin = 0, tmax = 1000, \
                      tstep = 0.1,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
    """ 'function' is the function to be integrated.
    
    initial_condition must be a matrix with shape = (number of variables, number of nodes) 
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the function other than time should be passed.

    It returns 2 matrices 'xpoints' and 'ypoint  whose columns are the 
    
    different nodes and the rows denote time
 
    """

    num_steps = int((tmax-tmin)/tstep)
    num_nodes = len(initial_conditions[0])


    tpoints, xpoints, ypoints  = np.arange(tmin, tmax, tstep), \
                                          np.zeros( shape = (num_steps, num_nodes)), \
                                          np.zeros( shape = (num_steps, num_nodes)), \

    R = initial_conditions

    if method == 'runge_kutta2':
      print('Method of integration: ', method)
      for t in range(num_steps):
        xpoints[t][::] = R[0][::]
        ypoints[t][::] = R[1][::]
        k1 = tstep*function(R, t, **kwargs)
        k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
        R += k2
   
    elif method == 'euler':
      print('Method of integration: ', method)
      for t in range(num_steps):
        xpoints[t][::] = R[0][::]
        ypoints[t][::] = R[1][::]
        R += tstep*function(R, t, **kwargs)
        
        
    elif method == 'euler_maruyama':
      print('Method of integration: ', method)
      for t in range(num_steps):
        dW = np.random.randn(1, num_nodes)
        xpoints[t][::] = R[0][::]
        ypoints[t][::] = R[1][::]

        R += tstep*function(R, t, **kwargs) + noise_amp*dW
        

    return xpoints, ypoints



def integration_ode_4(function, initial_conditions, tmin = 0, tmax = 1000, \
                      tstep = 0.1,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
    """ 'function' is the function to be integrated.
    
    initial_condition must be a matrix with shape = (number of variables, number of nodes) 
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the function other than time should be passed.

    It returns 2 matrices 'xpoints' and 'ypoint  whose columns are the 
    
    different nodes and the rows denote time
 
    """

    num_steps = int((tmax-tmin)/tstep)
    num_nodes = len(initial_conditions[0])


    zpoints  = np.zeros( shape = (num_steps, num_nodes), dtype = np.complex128)
                                       

    R = initial_conditions


    if method == 'runge_kutta2':
      print('Method of integration: ', method)
      for t in range(num_steps):
        zpoints[t][::] = R
        k1 = tstep*function(R, t, **kwargs)
        k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
        R += k2
   
    elif method == 'euler':
      print('Method of integration: ', method)
      for t in range(num_steps):
        zpoints[t][::] = R[0, :]
        z = function(R, t,  **kwargs)
        R[1:, :] = z[0:len(R) - 1, :]
        R[0, :] += tstep*z[0, :]
        
    elif method == 'euler_maruyama':
      print('Method of integration: ', method)
      for t in range(num_steps):
        dW = (np.random.randn(num_nodes) + 1j*np.random.randn(num_nodes))

        zpoints[t][::] = R[0, :]
        z = function(R, t,  **kwargs)
        R[1:, :] = z[0:-1, :]
        R[0, :] += tstep*z[0, :] + noise_amp*dW
        
      
        



    return zpoints


def integration_ode_5(function, initial_conditions, tmin = 0, tmax = 1000, \
                      tstep = 0.1,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
    """ 'function' is the function to be integrated.
    
    initial_condition must be a matrix with shape = (number of variables, number of nodes) 
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the function other than time should be passed.

    It returns 2 matrices 'xpoints' and 'ypoint  whose columns are the 
    
    different nodes and the rows denote time
 
    """

    num_steps = int((tmax-tmin)/tstep)
    num_nodes = len(initial_conditions[0])


    zpoints  = np.zeros( shape = (num_steps, num_nodes), dtype = np.complex64)
                                       

    R = initial_conditions


    if method == 'runge_kutta2':
      print('Method of integration: ', method)
      for t in range(num_steps):
        zpoints[t][::] = R
        k1 = tstep*function(R, t, **kwargs)
        k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
        R += k2
   
    elif method == 'euler':
      print('Method of integration: ', method)
      for t in range(num_steps):
        zpoints[t][::] = R
        R += tstep*function(R, t, **kwargs)
        
    elif method == 'euler_maruyama':
      print('Method of integration: ', method)
      dW = (np.random.randn(num_steps, num_nodes) + 1j*np.random.randn(num_steps, num_nodes))

      for t in range(num_steps):
        
        zpoints[t][::] = R
        R += tstep*function(R, t, **kwargs) + noise_amp*dW[t]
        

    return zpoints


def integration_ode_6(function, initial_conditions, tmin = 0, tmax = 1000, \
                      tstep = 0.1,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
    """ 'function' is the function to be integrated.
    
    initial_condition must be a matrix with shape = (number of variables, number of nodes) 
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the function other than time should be passed.

    It returns 2 matrices 'xpoints' and 'ypoint  whose columns are the 
    
    different nodes and the rows denote time
 
    """

    num_steps = int((tmax-tmin)/tstep)
    num_nodes = len(initial_conditions[0])


    zpoints  = np.zeros( shape = (num_steps, num_nodes), dtype = 'F')
                                 

    R = initial_conditions



    if method == 'runge_kutta2':
      print('Method of integration: ', method)
      for t in range(num_steps):
        zpoints[t][::] = R
        k1 = tstep*function(R, t, **kwargs)
        k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
        R += k2
   
    elif method == 'euler':
      print('Method of integration: ', method)
      for t in range(num_steps):
        zpoints[t][::] = R
        R += tstep*function(R, t, **kwargs)
        
    elif method == 'euler_maruyama':
      print('Method of integration: ', method)
      dW = np.random.randn(num_steps, num_nodes) + 1j*np.random.randn(num_steps, num_nodes)
      dW = np.array(dW, dtype = 'F')
      
      
      for t in range(num_steps):
        zpoints[t][::] = R
        R += tstep*function(R, t, **kwargs) + noise_amp*dW[t]
        

    return zpoints





def integration_ode_7(function, initial_conditions, tmin = 0, tmax = 1000, \
                      tstep = 0.1,  method = 'runge_kutta2', noise_amp = None, **kwargs):
    
    """ 'function' is the function to be integrated.
    
    initial_condition must be a matrix with shape = (number of variables, number of nodes) 
    
    tmin (tmax) is the initial (final) time of integration

    tstep: step of integration

    method: method of integration.

    In **kwargs the parameters of the function other than time should be passed.

    It returns 2 matrices 'xpoints' and 'ypoint  whose columns are the 
    
    different nodes and the rows denote time
 
    """

    num_steps = int((tmax-tmin)/tstep)
    num_nodes = len(initial_conditions[0])


    zpoints  = np.zeros( shape = (num_steps, num_nodes), dtype = 'F')
                                 

    R = initial_conditions


    if ('kinetic' not in kwargs) or (kwargs['kinetic'] is None):
        if method == 'runge_kutta2':
          print('Method of integration: ', method)
          for t in range(num_steps):
            zpoints[t][::] = R
            k1 = tstep*function(R, t, **kwargs)
            k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
            R += k2
       
        elif method == 'euler':
          print('Method of integration: ', method)
          for t in range(num_steps):
            zpoints[t][::] = R
            R += tstep*function(R, t, **kwargs)
            
        elif method == 'euler_maruyama':
          print('Method of integration: ', method)
          dW = np.random.randn(num_steps, num_nodes) + 1j*np.random.randn(num_steps, num_nodes)
          dW = np.array(dW, dtype = 'F')
          
          
          for t in range(num_steps):
            zpoints[t][::] = R

            R += tstep*function(R, t, **kwargs) + noise_amp*dW[t]
    
    else:
        
        if type(kwargs['kinetic']) is tuple:

            num_kinetic_param = len(kwargs['kinetic'])
            

            kinetic_parameter_1 = kwargs[kwargs['kinetic'][0]]
            kinetic_parameter_2 = kwargs[kwargs['kinetic'][1]]
            
  
            
            if method == 'runge_kutta2':
              print('Method of integration: ', method)
              for t in range(num_steps):
                kwargs[kwargs['kinetic']] = kinetic_parameter[t]
                
                zpoints[t][::] = R
                k1 = tstep*function(R, t, **kwargs)
                k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
                R += k2
           
            elif method == 'euler':
              print('Method of integration: ', method)
              for t in range(num_steps):
                kwargs[kwargs['kinetic']] = kinetic_parameter[t]
                
                zpoints[t][::] = R
                R += tstep*function(R, t, **kwargs)
                
            elif method == 'euler_maruyama':
              print('Method of integration: ', method)
              dW = np.random.randn(num_steps, num_nodes) + 1j*np.random.randn(num_steps, num_nodes)
              dW = np.array(dW, dtype = 'F')
              
              
              
              for t in range(num_steps):
                  kwargs[kwargs['kinetic'][0]] = kinetic_parameter_1[t]
                  kwargs[kwargs['kinetic'][1]] = kinetic_parameter_2[:, t]
                  
                  zpoints[t][::] = R
                  R += tstep*function(R, t, **kwargs) + noise_amp*dW[t]



        else:
            kinetic_parameter = kwargs[kwargs['kinetic']]
               
            if method == 'runge_kutta2':
              print('Method of integration: ', method)
              for t in range(num_steps):
                kwargs[kwargs['kinetic']] = kinetic_parameter[t]
                
                zpoints[t][::] = R
                k1 = tstep*function(R, t, **kwargs)
                k2 = tstep*function(R + 0.5*k1, t + 0.5, **kwargs)
                R += k2
           
            elif method == 'euler':
              print('Method of integration: ', method)
              for t in range(num_steps):
                kwargs[kwargs['kinetic']] = kinetic_parameter[t]
                
                zpoints[t][::] = R
                R += tstep*function(R, t, **kwargs)
                
            elif method == 'euler_maruyama':
              print('Method of integration: ', method)
              dW = np.random.randn(num_steps, num_nodes) + 1j*np.random.randn(num_steps, num_nodes)
              dW = np.array(dW, dtype = 'F')
              
              
              
              for t in range(num_steps):
                  if len(kinetic_parameter.shape) == 1:
                      kwargs[kwargs['kinetic']] = kinetic_parameter[t]
                  else:
                      kwargs[kwargs['kinetic']] = kinetic_parameter[:, t]
                
                  zpoints[t][::] = R
                  R += tstep*function(R, t, **kwargs) + noise_amp*dW[t]



    return zpoints

