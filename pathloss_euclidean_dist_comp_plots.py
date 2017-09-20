import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import vtovosm as vtv

file_format = 'pgf'
plt.rcParams["figure.figsize"] = (5, 3)
plt.rcParams['text.usetex'] = True
# plt.rcParams['lines.linewidth'] = 3
plt.rcParams['pgf.rcfonts'] = False
plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.style.use('ggplot')

densities = [10, 20, 30, 40, 50, 60, 70, 80, 120, 160]
count_veh = 337  # from results_all.keys() = [48, 96, 145, 193, 241, 289, 337, 386, 578, 771]
dir_out = os.path.join('images', 'viriyasitavat_comparison_pathloss')
overwrite = False

results_euclidean = vtv.utils.load('results/viriyasitavat_comparison_uniform/result_analysis.pickle.xz')
results_pl_los = vtv.utils.load('results/viriyasitavat_comparison_uniform_pathloss/result_max_from_los_analysis.pickle.xz')
results_pl_olos = vtv.utils.load('results/viriyasitavat_comparison_uniform_pathloss/result_max_from_olos_analysis.pickle.xz')
results_pl_nlos = vtv.utils.load('results/viriyasitavat_comparison_uniform_pathloss/result_max_from_nlos_analysis.pickle.xz')
results_pl_mean = vtv.utils.load('results/viriyasitavat_comparison_uniform_pathloss/result_max_from_mean_analysis.pickle.xz')

os.makedirs(dir_out, exist_ok=True)

mean_pl_los = dict()
mean_pl_olos = dict()
mean_pl_nlos = dict()
mean_pl_mean = dict()
mean_euclidean = dict()
for key in results_euclidean.keys():
    arr = np.array(results_pl_los[key]['net_connectivities'])
    mean_pl_los[key] = np.mean(arr[:,0])

    arr = np.array(results_pl_olos[key]['net_connectivities'])
    mean_pl_olos[key] = np.mean(arr[:,0])
    
    arr = np.array(results_pl_nlos[key]['net_connectivities'])
    mean_pl_nlos[key] = np.mean(arr[:,0])
    
    arr = np.array(results_pl_mean[key]['net_connectivities'])
    mean_pl_mean[key] = np.mean(arr[:,0])
    
    arr = np.array(results_euclidean[key]['net_connectivities'])
    mean_euclidean[key] = np.mean(arr)

filename = 'net_con_vs_veh_dens.' + file_format
path_out = os.path.join(dir_out, filename)
plt.figure()
plt.plot(densities, np.array(sorted(mean_euclidean.items()))[:,1], label='Euclidean distance')
plt.plot(densities, np.array(sorted(mean_pl_los.items()))[:,1], label='Pathloss (max from LOS)')
plt.plot(densities, np.array(sorted(mean_pl_nlos.items()))[:,1], label='Pathloss (max from NLOS)')
plt.plot(densities, np.array(sorted(mean_pl_olos.items()))[:,1], label='Pathloss (max from OLOS)')
plt.plot(densities, np.array(sorted(mean_pl_mean.items()))[:,1], label='Pathloss (max from mean)')
plt.xlim((min(densities), max(densities)))
plt.grid(True)
plt.xlabel(r'Vehicle density $[1/km^2]$')
plt.ylabel('Average network connectivity $\overline{NC}$')
plt.legend()
plt.tight_layout(pad=0.25)
plt.savefig(path_out)
plt.close()
