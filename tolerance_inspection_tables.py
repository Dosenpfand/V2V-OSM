import logging
import os

import pandas as pd

import vtovosm as vtv

# Config
dir_out = os.path.join('images', 'tolerance_inspection')
overwrite = True

# Functions
def frame_merge(df1, df2):
    """Prepares the data frame for latex output"""

    out = df1.copy()
    out['ratio_run'] = out['run_time_w'] / out['run_time_wo']
    out = out[['count_vehs', 'count_con_tot', 'ratio_con_diff', 'ratio_run']]
    out['ratio_diff_5'] = df2['ratio_con_diff']
    out['ratio_run_5'] = df2['run_time_w'] / df2['run_time_wo']
    out.columns = ['Vehicle count', 'Connection count', 'Error ratio - 1 m', 'Time ratio - 1 m', 'Error ratio - 5 m', 'Time ratio - 5 m',]

    return out

# Setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.info('Loading files')
os.makedirs(dir_out, exist_ok=True)
res_1_nyc = vtv.utils.load('results/tolerance_inspection_1m/tolerance_comparison_upperwestside.pickle.xz')
res_5_nyc = vtv.utils.load('results/tolerance_inspection_5m/tolerance_comparison_upperwestside.pickle.xz')
res_1_vie = vtv.utils.load('results/tolerance_inspection_1m/tolerance_comparison_neubau.pickle.xz')
res_5_vie = vtv.utils.load('results/tolerance_inspection_5m/tolerance_comparison_neubau.pickle.xz')

res_1_nyc_df = pd.DataFrame(res_1_nyc)
res_5_nyc_df = pd.DataFrame(res_5_nyc)
res_1_vie_df = pd.DataFrame(res_1_vie)
res_5_vie_df = pd.DataFrame(res_5_vie)

# Write table for NYC
filename = 'nyc.tex'
path_out = os.path.join(dir_out, filename)

if os.path.isfile(path_out) and not overwrite:
    logging.warning('file {} already exists. Skipping'.format(filename))
else:
    df = frame_merge(res_1_nyc_df, res_5_nyc_df)
    with open(path_out, 'w') as file:
        df.to_latex(buf=file, index=False, float_format='%.3g')
    logging.info('Saved {}'.format(filename))

# Write table for Vienna
filename = 'vie.tex'
path_out = os.path.join(dir_out, filename)

if os.path.isfile(path_out) and not overwrite:
    logging.warning('file {} already exists. Skipping'.format(filename))
else:
    df = frame_merge(res_1_vie_df, res_5_vie_df)
    with open(path_out, 'w') as file:
        df.to_latex(buf=file, index=False, float_format='%.3g')
    logging.info('Saved {}'.format(filename))
