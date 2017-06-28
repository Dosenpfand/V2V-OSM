import logging
import os

import matplotlib.pyplot as plt

import vtovosm as vtv

# Config
plt.rcParams["figure.figsize"] = (8, 5)
count_veh = 362  # from [121, 362, 241]
dir_out = os.path.join('images', 'speed_and_tls_cycle_impact')
overwrite = True
show_titles = False

# Setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.info('Loading files')
os.makedirs(dir_out, exist_ok=True)
res_45s_10mps = vtv.utils.load('results/speed_and_tls_cycle_impact/result_45s10mps_analysis.pickle.xz')
res_45s_15mps = vtv.utils.load('results/speed_and_tls_cycle_impact/result_45s15mps_analysis.pickle.xz')
res_90s_15mps = vtv.utils.load('results/speed_and_tls_cycle_impact/result_90s15mps_analysis.pickle.xz')

# Plot link duration pmf for one specific vehicle density and different maximum vehicle speeds
filename = 'link_dur_pmf_speed_unweighted.pdf'
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
    plt.xlabel('Link duration [s]')
    plt.ylabel('Probability')
    if show_titles:
        plt.title('Unweighted Link Duration Distribution - Maximum Speed Impact')
    plt.legend()
    plt.savefig(path_out)
    logging.info('Saved {}'.format(filename))

# Plot link duration pmf for one specific vehicle density and different TLS cycle times
filename = 'link_dur_pmf_tls_unweighted.pdf'
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
    plt.xlabel('Link duration [s]')
    plt.ylabel('Probability')
    if show_titles:
        plt.title('Unweighted Link Duration Distribution - TLS Cycle Time Impact')
    plt.legend()
    plt.savefig(path_out)
    logging.info('Saved {}'.format(filename))

# Plot weighted link duration pmf for one specific vehicle density and different maximum vehicle speeds
filename = 'link_dur_pmf_speed_weighted.pdf'
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
    plt.xlabel('Link duration [s]')
    plt.ylabel('Probability')
    if show_titles:
        plt.title('Weighted Link Duration Distribution - Maximum Speed Impact')
    plt.legend()
    plt.savefig(path_out)
    logging.info('Saved {}'.format(filename))

# Plot weighted link duration pmf for one specific vehicle density and different TLS cycle times
filename = 'link_dur_pmf_tls_weighted.pdf'
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
    plt.xlabel('Link duration [s]')
    plt.ylabel('Probability')
    if show_titles:
        plt.title('Weighted Link Duration Distribution - TLS Cycle Time Impact')
    plt.legend()
    plt.savefig(path_out)
    logging.info('Saved {}'.format(filename))
