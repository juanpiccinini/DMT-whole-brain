from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def from_aal_to_deco(arr):
    deco_order = np.zeros_like(arr)
    deco_order[0:45] = arr[0:90:2]
    deco_order[-1:44:-1] = arr[1:90:2]
    
    return deco_order

def kinetic(scale, alpha, beta, t):
  fun = (t**(alpha)*np.exp(-t/beta))
  normalization = np.sum(fun)

  return scale*fun/normalization



data_empirical = loadmat(r"empirical_observables_DMT_experiment.mat")
data_SC = loadmat(r"SC_data.mat")
SC = data_SC.get('SC', None)    #structure connectivity matrix

df_new = pd.read_csv(r"df_susceptibility.csv", index_col='index')
df_new.drop(columns = ['Unnamed: 0'], inplace = True)

condition_emp = 'DMT'


FCD_empirical = data_empirical.get(f'FCD_{condition_emp}', None)   
   

tStep = 0.1
#attributes of the Simulation Class (Hopf in this case)
t_final = 1800
pre_time = 480      #8 minutes in resting state
thermalization_time = 120


time = np.arange(0, t_final, tStep)
dose_time = np.arange(0, t_final - (pre_time + thermalization_time), tStep)

data_rsn = loadmat(r"membership_rsn.mat")
rsn_member = data_rsn.get('rsn_member')    

receptor_density_df_1 = pd.read_csv(r"5HT2a_alt_hc19_savli.csv", header = None)
receptor_density_aal_1 = receptor_density_df_1.to_numpy().squeeze()
receptor_density_aal_1 = receptor_density_aal_1[0:90]   #only get the 90 nodes we want
receptor_density_1 = from_aal_to_deco(receptor_density_aal_1)

receptor_density_df_2 = pd.read_csv(r"5HT2a_cimbi_hc29_beliveau.csv", header = None)
receptor_density_aal_2 = receptor_density_df_2.to_numpy().squeeze()
receptor_density_aal_2 = receptor_density_aal_2[0:90]   #only get the 90 nodes we want
receptor_density_2 = from_aal_to_deco(receptor_density_aal_2)

receptor_density_df_3 = pd.read_csv(r"5HT2a_mdl_hc3_talbot.csv", header = None)
receptor_density_aal_3 = receptor_density_df_3.to_numpy().squeeze()
receptor_density_aal_3 = receptor_density_aal_3[0:90]   #only get the 90 nodes we want
receptor_density_3 = from_aal_to_deco(receptor_density_aal_3)

receptor_density = ((19/51)*receptor_density_1 + (29/51)*receptor_density_2 + (3/51)*receptor_density_3)

rsn_receptors = rsn_member*receptor_density.reshape(-1, 1)
rsn_receptors_net = rsn_receptors.sum(axis = 0)/np.count_nonzero(rsn_receptors, axis = 0)

rsn_receptors_net = rsn_receptors_net/rsn_receptors_net.sum()



redes_df = {'RSN': ['Vis', 'ES', 'Aud', 'SM',  'DM', 'EC'], '5HT2a Density': rsn_receptors_net}
df_net_names = pd.DataFrame(redes_df)

# Order the bars by 'Value' in ascending order
ordered_df = df_net_names.sort_values(by='5HT2a Density', ascending = True)


best_scale_dmt = 159
best_beta_dmt = 284

best_scale_pcb = 65.6
best_beta_pcb = 588.0  

stim_net = np.arange(6)
#F_ext_all = np.arange(0, 0.1, 0.005)[0:5]
F_ext_all = np.arange(0, 0.0175, 0.0025)[0:-1]

a_constant_dmt = -1*kinetic(best_scale_dmt, 1, best_beta_dmt, dose_time)[::250] + 0.07
a_constant_pcb = -1*kinetic(best_scale_pcb, 1, best_beta_pcb, dose_time)[::250] + 0.07

dose_time_downsample = dose_time[::250]



sns.set_style("darkgrid")
sns.lineplot(data=df_new[(df_new.condition == 'DMT' ) & (df_new.F_ext == 0.01 )], x="time", y="susceptibility_per_node",  errorbar = 'sd', hue = 'net_name', palette = 'bright', hue_order = ordered_df.RSN.array)
plt.ylabel('Susceptibility per Node')
plt.xlabel('Time (s)')
plt.xlim(0, 1200)
plt.ylim(0, 1.4)
plt.legend(title='RSN')
plt.savefig(r'susceptibility_DMT.png', dpi=1200)
plt.close()

sns.set_style("darkgrid")
sns.lineplot(data=df_new[(df_new.condition == 'PCB' ) & (df_new.F_ext == 0.01 )], x="time", y="susceptibility_per_node",  errorbar = 'sd', hue = 'net_name', palette = 'bright', hue_order = ordered_df.RSN.array)
plt.ylabel('Susceptibility per Node')
plt.xlabel('Time (s)')
plt.xlim(0, 1200)
plt.ylim(0, 1.4)
plt.legend(title='RSN')
plt.savefig(r'susceptibility_PCB.png', dpi=1200)
plt.close()


#%%


"""plots peak of the reactivity"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_fit_2 = pd.read_csv('df_susceptibility_peak.csv', index_col= 'Unnamed: 0')


red_blue_palette = sns.color_palette("Set1", 2)
custom_palette = [red_blue_palette[1], red_blue_palette[0]]

sns.set_style("darkgrid")
# Create a figure with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Adjust the aspect ratio
fig.subplots_adjust(wspace=0.4, hspace=0.3)

count = 0
nombre_redes = [ 'Aud', 'DM', 'SM','ES',  'EC', 'Vis']
# Plot violin plots in each subplot
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        sns.violinplot(data = df_fit_2[(df_fit_2.F_ext.isin([0.005, 0.01, 0.015])) & (df_fit_2.net_name == nombre_redes[count])], x = 'F_ext', y = 'max_value', hue = 'condition', palette = custom_palette, ax=ax, label = False)
        ax.legend(title = f'{nombre_redes[count]}',title_fontsize=20, loc = 'upper left')
        ax.set_ylabel('$\chi_{max}$')
        ax.set_xlabel('$F_{ext}$')
        ax.set_ylim(0.0, 1.6)
        for k in range(len(np.unique([0.005, 0.01, 0.015])) - 1):
            ax.axvline(k + 0.5, color='grey', lw=1.5, linestyle='--' )
        count += 1


plt.savefig(fr'maximum_Fext_redes.png', dpi=1200)
plt.close()





#%%
from statsmodels.stats.weightstats import DescrStatsW

def sample_group(group):
    return group.sample(n=100, replace = True)

def sample_group_condition(df, x_var):
    sample = df.groupby(x_var, group_keys=False).apply(sample_group)

    return sample



df_fit_3 = pd.read_csv("df_susceptibility_delta_peak.csv")


F_ext_all = np.arange(0, 0.0175, 0.0025)[0:-1]
num_boots_simulations = 1000

df_correlation =  dict()
df_correlation['condition'] = []
df_correlation['F_ext'] = []
df_correlation['corr_maximum'] = []


for fuerza_index, fuerza in enumerate(F_ext_all):
    df_fit_DMT = df_fit_3[(df_fit_3.condition == 'resta' ) & (df_fit_3.F_ext== fuerza)].copy()


    errores_DMT_max = df_fit_DMT.groupby('net_name')['max_value'].var().to_dict()

    df_fit_DMT['var_max'] = df_fit_DMT['net_name'].map(errores_DMT_max)


    for num_boots in range(num_boots_simulations):
        sample_DMT = sample_group_condition(df_fit_DMT, 'net_name', )

        max_value_DMT = sample_DMT['max_value']
        recep_density_DMT = sample_DMT['5HT2a Density']

        
        skewness_DMT = sample_DMT['skewness']

  
        peso_DMT_max = 1/sample_DMT['var_max'].values
        prueba_DMT = pd.concat([max_value_DMT, recep_density_DMT], axis = 1)
        wcorr_DMT = DescrStatsW(prueba_DMT, weights=peso_DMT_max)


        

        df_correlation['condition'].append('resta')
        df_correlation['F_ext'].append(fuerza)
        df_correlation['corr_maximum'].append(wcorr_DMT.corrcoef[0, 1])
 
        
df_correlation = pd.DataFrame(df_correlation)



plt.figure(figsize = (6, 5))
sns.set_style("dark")
sns.histplot(data=df_correlation[(df_correlation.F_ext == 0.01) & ((df_correlation.condition == 'resta'))], x= 'corr_maximum', color = 'purple', bins = 50)
plt.xlabel(r"$\rho$")
plt.savefig(r'corr_difference_01.png', dpi=1200)
plt.close()


sns.set_style("darkgrid")
# Create the regplot
ax1 = sns.regplot(x='5HT2a Density', y="max_value", data= df_fit_3[(df_fit_3.F_ext== 0.01) & (df_fit_3.condition == 'resta')], color = 'purple', 
            x_estimator=np.mean, scatter_kws={'s':5}, x_ci = None)


x = df_fit_3[(df_fit_3.F_ext== 0.01) & (df_fit_3.condition == 'resta')].groupby('net_name')['5HT2a Density'].mean().values
y_DMT = df_fit_3[(df_fit_3.F_ext== 0.01) & (df_fit_3.condition == 'resta')].groupby('net_name')['max_value'].agg(['mean', 'std'])

# Customize error bars
for i in range(len(x)):
    ax1.errorbar(x[i], y_DMT['mean'].values[i], yerr = y_DMT['std'].values[i], capsize=2, color='purple', lw=1)
plt.ylabel('$\Delta \chi_{max}$')
plt.xlabel('5HT2a Density')
plt.ylim(0.2, 1.4)
#plt.legend(title='Condition', loc = 'lower right')
plt.savefig(r'corr_delta_maximum.png', dpi=1200)
plt.close()


sns.set_style("darkgrid")
x= df_correlation[(df_correlation.condition == 'resta') & (df_correlation.F_ext.isin(F_ext_all[1::]))].F_ext.unique()


y = df_correlation[(df_correlation.condition == 'resta') & (df_correlation.F_ext.isin(F_ext_all[1::]))].groupby('F_ext').corr_maximum.agg(['mean', 'std'])
# Plot only error bars

plt.plot(x, y['mean'].values, '-', linewidth=2, markersize=None, color = 'purple')
plt.errorbar(x, y['mean'].values, yerr=y['std'].values, fmt='none', marker='', capsize=4, ecolor='purple')

plt.xlim(0.002, 0.016)
plt.xlabel(r"$F_{0}$")
plt.ylabel(r"$\rho$")
plt.savefig(r'corr_max_diff_vs_Fext.png', dpi=1200)
plt.close()