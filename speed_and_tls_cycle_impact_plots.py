import logging
import os

import matplotlib.pyplot as plt

import vtovosm as vtv

# Config
file_format = 'pgf'
plt.rcParams["figure.figsize"] = (5, 3)
plt.rcParams['text.usetex'] = True
# plt.rcParams['lines.linewidth'] = 3
plt.rcParams['pgf.rcfonts'] = False
plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.style.use('ggplot')

counts_veh = [121, 241, 362]
dir_out = os.path.join('images', 'speed_and_tls_cycle_impact')
overwrite = True

# Setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
os.makedirs(dir_out, exist_ok=True)

logging.info('Loading files')
res_45s_10mps = vtv.utils.load('results/speed_and_tls_cycle_impact/result_45s10mps_analysis.pickle.xz')
res_45s_15mps = vtv.utils.load('results/speed_and_tls_cycle_impact/result_45s15mps_analysis.pickle.xz')
res_90s_15mps = vtv.utils.load('results/speed_and_tls_cycle_impact/result_90s15mps_analysis.pickle.xz')

for count_veh in counts_veh:

    # Plot link duration pmf for one specific vehicle density and different maximum vehicle speeds
    filename = 'link_dur_pmf_speed_unweighted_{:d}.{}'.format(count_veh, file_format)
    path_out = os.path.join(dir_out, filename)

    link_durations_45s_10mps = res_45s_10mps[count_veh]['link_durations'].durations_con
    bins_45s_10_mps = int(max(link_durations_45s_10mps) - min(link_durations_45s_10mps))
    link_durations_45s_15mps = res_45s_15mps[count_veh]['link_durations'].durations_con
    bins_45s_15_mps = int(max(link_durations_45s_15mps) - min(link_durations_45s_15mps))
    link_durations_90s_15mps = res_90s_15mps[count_veh]['link_durations'].durations_con
    bins_90s_15_mps = int(max(link_durations_90s_15mps) - min(link_durations_90s_15mps))

    if os.path.isfile(path_out) and not overwrite:
        logging.warning('file {} already exists. Skipping'.format(filename))
    else:
        plt.figure()
        hist = plt.hist(link_durations_45s_15mps, bins=bins_45s_15_mps, normed=True, alpha=0.6,
                        label='15 m/s max. speed')
        hist = plt.hist(link_durations_45s_10mps, bins=bins_45s_10_mps, normed=True, alpha=0.6,
                        label='10 m/s max. speed')
        plt.xlim((0, 100))
        plt.grid(True)
        plt.xlabel('Unique link duration $T_{l,u}$ [s]')
        plt.ylabel('Distribution $p(T_{l,u})$')
        plt.legend()
        plt.tight_layout(pad=0.25)
        plt.savefig(path_out)
        logging.info('Saved {}'.format(filename))

    # Plot link duration pmf for one specific vehicle density and different TLS cycle times
    filename = 'link_dur_pmf_tls_unweighted_{:d}.{}'.format(count_veh, file_format)
    path_out = os.path.join(dir_out, filename)

    if os.path.isfile(path_out) and not overwrite:
        logging.warning('file {} already exists. Skipping'.format(filename))
    else:
        plt.figure()
        hist = plt.hist(link_durations_45s_15mps, bins=bins_45s_15_mps, normed=True, alpha=0.6,
                        label='45 s TLS cycle time')
        hist = plt.hist(link_durations_90s_15mps, bins=bins_90s_15_mps, normed=True, alpha=0.6,
                        label='90 s TLS cycle time')
        plt.xlim((0, 100))
        plt.grid(True)
        plt.xlabel('Unique link duration $T_{l,u}$ [s]')
        plt.ylabel('Distribution $p(T_{l,u})$')
        plt.legend()
        plt.tight_layout(pad=0.25)
        plt.savefig(path_out)
        logging.info('Saved {}'.format(filename))

    # Plot weighted link duration pmf for one specific vehicle density and different maximum vehicle speeds
    filename = 'link_dur_pmf_speed_weighted_{:d}.{}'.format(count_veh, file_format)
    path_out = os.path.join(dir_out, filename)

    if os.path.isfile(path_out) and not overwrite:
        logging.warning('file {} already exists. Skipping'.format(filename))
    else:
        plt.figure()
        hist = plt.hist(link_durations_45s_15mps, bins=bins_45s_15_mps, normed=True, weights=link_durations_45s_15mps,
                        alpha=0.6, label='15 m/s max. speed')
        hist = plt.hist(link_durations_45s_10mps, bins=bins_45s_10_mps, normed=True, weights=link_durations_45s_10mps,
                        alpha=0.6, label='10 m/s max. speed')
        plt.xlim((0, 100))
        plt.grid(True)
        plt.xlabel('Total link duration $T_{l,t}$ [s]')
        plt.ylabel('Distribution $p(T_{l,t})$')
        plt.legend()
        plt.tight_layout(pad=0.25)
        plt.savefig(path_out)
        logging.info('Saved {}'.format(filename))

    # Plot weighted link duration pmf for one specific vehicle density and different TLS cycle times
    filename = 'link_dur_pmf_tls_weighted_{:d}.{}'.format(count_veh, file_format)
    path_out = os.path.join(dir_out, filename)

    if os.path.isfile(path_out) and not overwrite:
        logging.warning('file {} already exists. Skipping'.format(filename))
    else:
        plt.figure()
        hist = plt.hist(link_durations_45s_15mps, bins=bins_45s_15_mps, normed=True, weights=link_durations_45s_15mps,
                        alpha=0.6, label='45 s TLS cycle time')
        hist = plt.hist(link_durations_90s_15mps, bins=bins_90s_15_mps, normed=True, weights=link_durations_90s_15mps,
                        alpha=0.6, label='90 s TLS cycle time')
        plt.xlim((0, 100))
        plt.grid(True)
        plt.xlabel('Unique link duration $T_{l,t}$ [s]')
        plt.ylabel('Distribution $p(T_{l,t})$')
        plt.legend()
        plt.tight_layout(pad=0.25)
        plt.savefig(path_out)
        logging.info('Saved {}'.format(filename))
