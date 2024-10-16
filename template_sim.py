from scipy.io import loadmat, savemat
import numpy as np
from Simulations import *
from scipy.stats import gamma
import pandas as pd


def kinetic(scale, alpha, beta, t):
  fun = (t**(alpha)*np.exp(-t/beta))
  normalization = np.sum(fun)

  return scale*fun/normalization

#load all needed data for the Simulation 
data = loadmat('SC_data.mat')
data_empirical = loadmat("empirical_observables_DMT_experiment.mat")

condition = 'DMT'

FCD_empirical = data_empirical.get(f'FCD_{condition}', None)    
FC_empirical = data_empirical.get(f'FC_{condition}', None)    
FCt_empirical = data_empirical.get(f'FCt_{condition}', None)

FCD_hist_empirical = (data_empirical.get(f'FCD_hist_{condition}', None)).squeeze()
Synchronization_empirical = data_empirical.get(f'Synchronization_{condition}', None)[0]
Metastability_empirical = data_empirical.get(f'Metastability_{condition}', None)[0]

SC = data.get('SC', None)    #structure connectivity matrix
numNodes = len(SC)  #number of nodes
K = np.sum(SC, axis = 0) # constant to use in the models (coupling term)

freq = (( data_empirical['freq_DMT'] + data_empirical['freq_PCB'] )/2).squeeze()



tStep = 0.1

SC = SC + 1j*np.zeros_like(SC)
SC = np.array(SC, dtype = 'F')


#attributes of the Simulation Class (Hopf in this case)

t_final = 1800
pre_time = 480      #8 minutes in resting state
thermalization_time = 120
sim = Hopf_simulation(t1= 0, t2=t_final, t_thermal = thermalization_time, rescale_time=2, t_step = tStep, num_nodes = numNodes, integration_method = 'euler_maruyama', filter_band = (0.01, 0.08))
sim.FCD_parameters(60, 40, 2)


time = np.arange(0, t_final, tStep)
dose_time = np.arange(0, t_final - (pre_time + thermalization_time), tStep)


num_experiments = 2

# #PCB
# scale = 66
# beta = 588

if condition == 'DMT':
    #DMT
    scale = 159
    beta = 284

elif condition == 'PCB':
    #PCB
    scale = 66
    beta = 588

G = 0.5

subjects = 15
num_wind = FCD_empirical.shape[0]
num_sup_diag_FCD = int((num_wind*num_wind - num_wind)/2)

FCD_all = []
FC_all = []
Coherence_all = []

for exper in range(num_experiments):



    a = np.ones(int(t_final/tStep))*(0.07)
    #a_distribution = gamma.pdf(dose_time, alpha, scale = 1/beta)
    
    if scale == 1:
        a_distribution = np.zeros_like(dose_time) + beta
    
    else:
        a_distribution = kinetic(scale, 1, beta, dose_time)
    
    
    a[int((pre_time + thermalization_time)/tStep)::] -=  a_distribution

    fcd = np.zeros(shape = (subjects, num_sup_diag_FCD))
    
    FCs = np.zeros(shape = (subjects, numNodes, numNodes))
    FCDs = np.zeros(shape = (subjects, len(FCD_empirical), len(FCD_empirical)))

    Cos_dist_all = np.empty(shape = (subjects,  840, numNodes, numNodes))

  
    for sub in range(subjects):
        print('subject:', sub)
        
        ic =  np.random.uniform(-0.1, 0.1, size = (1, numNodes)) + 1j*np.random.uniform(-0.1, 0.1, size = (1, numNodes))

        ic = np.array(ic, dtype = 'F')
        
        sim.model_parameters(a = a,  M = SC, frequencies = freq, constant = K , G = G, noise_amp = 0.05*np.sqrt(tStep), kinetic = 'a')


        sim.initial_conditions(ic)
        sim.run_sim() 
        fcd[sub] = sim.FCD[np.triu_indices(sim.FCD.shape[0], k = 1)]
        FCs[sub] = sim.FC
        FCDs[sub] = sim.FCD
        
        Cos_dist = Cos_distance(sim.phase)
        Cos_dist_all[sub] = Cos_dist

        
    
    fisher_FC = np.arctanh(FCs)
    fisher_FC_mean = np.mean(fisher_FC, axis = 0)
    
    FC_mean = np.tanh(fisher_FC_mean)
    
    fisher_FCD = np.arctanh(FCDs)
    fisher_FCD_mean = np.mean(fisher_FCD, axis = 0)
    
    FCD_mean = np.tanh(fisher_FCD_mean)
    
    FCD_all.append(FCD_mean)
    FC_all.append(FC_mean)
    Coherence_all.append(Cos_dist_all.mean(axis = 0))

    
FCD_all = np.array(FCD_all)
FC_all = np.array(FC_all)
Coherence_all = np.array(Coherence_all)





