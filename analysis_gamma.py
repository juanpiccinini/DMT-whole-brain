from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def kinetic(scale, alpha, beta, t):
  fun = (t**(alpha)*np.exp(-t/beta))
  normalization = np.sum(fun)

  return scale*fun/normalization



best_df = pd.read_csv(r"simulated_data.csv")

save_location = r"D:\Juan\Facultad\Doctorado\Submmiting\code"

dmt_matrix = np.load(save_location + os.sep + r"DMT_matrix.npy")
pcb_matrix = np.load(save_location + os.sep + r"PCB_matrix.npy")

scale_all = np.arange(0, 225, 5)
beta_all = np.arange(20, 920, 20)


red_blue_palette = sns.color_palette("Set1", 2)
custom_palette = [red_blue_palette[1], red_blue_palette[0]]


sns.set_style("darkgrid")
sns.jointplot(data=best_df, x="beta", y="scale", hue= 'condition', palette=custom_palette, s = 70, alpha = 0.7)
sns.scatterplot(x = [284, 588], y = [159.3, 65.6], hue=['DMT', 'PCB'], palette=custom_palette, s=210, legend=False)
plt.ylabel(r"$\lambda$")
plt.xlabel(r"$\beta$")
plt.xlim(0, 1000)
plt.ylim(0, 250)

plt.savefig( "parameters_distribution.png", dpi=1200)
plt.close()

time_total = np.arange(0, 1680, 0.1)
time_inyection = np.arange(0, 1200, 0.1)
time_previous = np.arange(0, 480, 0.1)

dmt_baseline, pcb_baseline = 0.07, 0.07


# Create a figure and axes
fig, ax = plt.subplots()

# Loop through each time series and plot it
for i in range(len(best_df)):
    if best_df.iloc[i].condition == 'DMT':
        escala = best_df.iloc[i].scale
        b = best_df.iloc[i].beta
        dmt_distribution = np.concatenate(([0]*len(time_previous), -kinetic(escala, 1, b, time_inyection))) + dmt_baseline
        ax.plot(time_total, dmt_distribution, color = custom_palette[0], alpha = 0.25)
    else:
        escala = best_df.iloc[i].scale
        b = best_df.iloc[i].beta    
        pcb_distribution = np.concatenate(([0]*len(time_previous), -kinetic(escala, 1, b, time_inyection))) + pcb_baseline
        ax.plot(time_total, pcb_distribution, color = custom_palette[1], alpha = 0.25)


best_parameters = best_df.groupby(by = 'condition').mean()

escala_dmt, escala_pcb = best_parameters.loc['DMT'].scale, best_parameters.loc['PCB'].scale
b_dmt, b_pcb = best_parameters.loc['DMT'].beta, best_parameters.loc['PCB'].beta

dmt_best_distribution = np.concatenate(([0]*len(time_previous), -kinetic(escala_dmt, 1, b_dmt, time_inyection))) + dmt_baseline
pcb_best_distribution = np.concatenate(([0]*len(time_previous), -kinetic(escala_pcb, 1, b_pcb, time_inyection))) + pcb_baseline

ax.plot(time_total, dmt_best_distribution, color = custom_palette[0], linewidth = 3, label = 'DMT')
ax.plot(time_total, pcb_best_distribution, color = custom_palette[1], linewidth =3, label = 'PCB')
#ax.set_xlim(400, 1680)
ax.set_ylim(0.035, 0.075)
# Add labels and title
ax.set_xlabel("Time (s)")
ax.set_ylabel("Bifurcation Parameter (a)")

# Add legend
ax.legend()
plt.savefig('gamma_distributions.png', dpi=1200)
plt.close()


sns.set_style("white")
dmt_matrix = np.array(dmt_matrix)
plt.imshow(dmt_matrix.mean(axis = 0), vmin = 0.15, vmax = 0.38, cmap = 'jet')
x_ticks_locations = np.arange(0, 45, 5)  # Custom tick locations
y_ticks_locations = np.arange(0, 45, 5)
y_ticks_labels = scale_all[::5]
x_ticks_labels = beta_all[::5]
plt.xticks(x_ticks_locations, x_ticks_labels)
plt.yticks(y_ticks_locations, y_ticks_labels)
plt.ylabel('\u03BB')	
plt.xlabel('\u03B2')
plt.colorbar()
plt.savefig('DMT_grid.png', dpi = 1200)
plt.close()

pcb_matrix = np.array(pcb_matrix)
plt.imshow(pcb_matrix.mean(axis = 0), vmin = 0.15, vmax = 0.38, cmap = 'jet')
x_ticks_locations = np.arange(0, 45, 5)  # Custom tick locations
y_ticks_locations = np.arange(0, 45, 5)
y_ticks_labels = scale_all[::5]
x_ticks_labels = beta_all[::5]
plt.xticks(x_ticks_locations, x_ticks_labels)
plt.yticks(y_ticks_locations, y_ticks_labels)
plt.ylabel('\u03BB')	
plt.xlabel('\u03B2')
plt.colorbar()
plt.savefig('PCB_grid.png', dpi = 1200)
plt.close()
