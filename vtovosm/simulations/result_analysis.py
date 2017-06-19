"""Provides functions to derive further results from simulation results"""

import logging
import multiprocessing as mp
import os
from itertools import repeat

import networkx as nx
import numpy as np

from .. import connection_analysis as con_ana
from .. import geometry as geom_o
from .. import network_parser as nw_p
from .. import osmnx_addons as ox_a
from .. import utils


def main(conf_path=None, scenario=None):
    """Main result analysis function"""

    # Load the configuration
    if conf_path is None:
        config = nw_p.params_from_conf()
        if scenario is None:
            config_scenario = nw_p.params_from_conf(in_key=config['scenario'])
        else:
            config_scenario = nw_p.params_from_conf(in_key=scenario)
            config['scenario'] = scenario
    else:
        config = nw_p.params_from_conf(config_file=conf_path)
        if scenario is None:
            config_scenario = nw_p.params_from_conf(in_key=config['scenario'], config_file=conf_path)
        else:
            config_scenario = nw_p.params_from_conf(in_key=scenario, config_file=conf_path)
            config['scenario'] = scenario

    if isinstance(config_scenario, (list, tuple)):
        raise RuntimeError('Multiple scenarios not supported. Use appropriate function')

    # Merge the two configurations
    config = nw_p.merge(config, config_scenario)

    # Sanitize config
    config = nw_p.check_fill_config(config)
    densities_veh = config['densities_veh']

    # Return if there is nothing to analyze
    if config['analyze_results'] is None:
        return

    loglevel = logging.getLevelName(config['loglevel'])
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Load street network
    time_start = utils.debug(None, 'Loading street network')
    net = ox_a.load_network(config['place'],
                            which_result=config['which_result'],
                            tolerance=config['building_tolerance'])
    graph_streets = net['graph_streets']
    utils.debug(time_start)

    # Convert vehicle densities to counts
    counts_veh = np.zeros(densities_veh.size, dtype=int)

    if config['density_type'] == 'length':
        street_lengths = geom_o.get_street_lengths(graph_streets)

    # Determine total vehicle count
    for idx, density_veh in enumerate(densities_veh):
        if config['density_type'] == 'absolute':
            counts_veh[idx] = int(density_veh)
        elif config['density_type'] == 'length':
            counts_veh[idx] = int(round(density_veh * np.sum(street_lengths)))
        elif config['density_type'] == 'area':
            area = net['gdf_boundary'].area
            counts_veh[idx] = int(round(density_veh * area))
        else:
            raise ValueError('Density type not supported')

    # Determine file paths
    filepaths_res = []
    filepaths_ana = []
    filepaths_ana_all = []
    for idx_count_veh, count_veh in enumerate(counts_veh):

        # Determine results path and check if it exists
        if config['results_file_prefix'] is not None:
            filename_prefix = utils.string_to_filename(config['results_file_prefix'])
        elif 'scenario' in config:
            filename_prefix = utils.string_to_filename(config['scenario'])
        else:
            filename_prefix = utils.string_to_filename(config['place'])

        file_name_res = '{}.{:d}.pickle.xz'.format(filename_prefix, count_veh)
        file_name_ana = '{}.{:d}_analysis.pickle.xz'.format(filename_prefix, count_veh)

        if config['results_file_dir'] is not None:
            file_dir = config['results_file_dir']
        else:
            file_dir = 'results'

        filepath_res = os.path.join(file_dir, file_name_res)
        filepath_ana = os.path.join(file_dir, file_name_ana)

        result_file_exists = os.path.isfile(filepath_res)
        if not result_file_exists:
            raise FileNotFoundError('Result file not found')

        analysis_file_exists = os.path.isfile(filepath_ana)
        if analysis_file_exists:
            if config['overwrite_result']:
                logging.warning('Analysis file already exists. Overwriting')
                filepaths_res.append(filepath_res)
                filepaths_ana.append(filepath_ana)
            else:
                logging.warning('Analysis file already exists. Skipping analysis')
        else:
            filepaths_res.append(filepath_res)
            filepaths_ana.append(filepath_ana)

        filepaths_ana_all.append(filepath_ana)

    logging.info('Starting analysis of results')

    if config['simulation_mode'] == 'parallel':
        multiprocess = True
    elif config['simulation_mode'] == 'sequential':
        multiprocess = False
    else:
        raise NotImplementedError('Mode not supported')

    # Iterate all result files
    for filepath_res, filepath_ana in zip(filepaths_res, filepaths_ana):
        # Analyze results
        analyze_single(filepath_res, filepath_ana, config['analyze_results'], multiprocess=multiprocess)

    logging.info('Merging all analysis results')
    analysis_results = {}
    for count_veh, filepath_ana in zip(counts_veh, filepaths_ana_all):
        analysis_results[count_veh] = utils.load(filepath_ana)

    file_name_ana = '{}_analysis.pickle.xz'.format(filename_prefix)
    filepath_ana = os.path.join(file_dir, file_name_ana)
    analysis_file_exists = os.path.isfile(filepath_ana)
    if analysis_file_exists:
        if config['overwrite_result']:
            logging.warning('Overwriting combined analysis file')
            utils.save(analysis_results, filepath_ana)
        else:
            logging.warning('Combined analysis file already exists. Not overwriting')
    else:
        utils.save(analysis_results, filepath_ana)

    return analysis_results


def analyze_single(filepath_res, filepath_ana, config_analysis, multiprocess=False):
    """Runs a single vehicle count analysis of a simulation result.
    Can be run in parallel"""

    # Load the connection results
    results_loaded = utils.load(filepath_res)
    matrices_cons = results_loaded['results']['matrices_cons']
    vehs = results_loaded['results']['vehs']

    # Check if given connection results are not empty
    if len(matrices_cons) == 0 or vehs[0].count == 1:
        logging.warning('Nothing to analyze. Skipping')
        utils.save(None, filepath_ana)
        return

    # Check if analysis to be performed is set
    if config_analysis is None:
        return

    # Prepare analysis
    time_start = utils.debug(None, 'Analyzing results')

    all_analysis = ['net_connectivities',
                    'path_redundancies',
                    'link_durations',
                    'connection_durations']

    if config_analysis == ['all'] or config_analysis == 'all':
        config_analysis = all_analysis

    if not set(config_analysis).issubset(set(all_analysis)):
        raise RuntimeError('Analysis not supported')

    graphs_cons = []
    for matrix_cons in matrices_cons:
        graphs_cons.append(nx.from_numpy_matrix(matrix_cons))

    analysis_result = {}

    if multiprocess:
        pool = mp.Pool()

    # Determine network connectivities
    if 'net_connectivities' in config_analysis:
        logging.info('Determining network connectivities')

        if multiprocess:
            net_connectivities = pool.map(con_ana.calc_net_connectivity, graphs_cons)
        else:
            net_connectivities = con_ana.calc_net_connectivities(graphs_cons)

        analysis_result['net_connectivities'] = net_connectivities

    # Determine path redundancies
    if 'path_redundancies' in config_analysis:
        logging.info('Determining path redundancies')

        if multiprocess:
            # TODO: merge into 1 np array (not list of arrays)
            path_redundancies = pool.starmap(con_ana.calc_center_path_redundancy, zip(graphs_cons, vehs))
        else:
            path_redundancies = con_ana.calc_center_path_redundancies(graphs_cons, vehs)

        analysis_result['path_redundancies'] = path_redundancies

    # Determine link durations
    if 'link_durations' in config_analysis:
        logging.info('Determining link durations')
        # TODO: Not yet parallelized
        link_durations = con_ana.calc_link_durations(graphs_cons)
        analysis_result['link_durations'] = link_durations

    # Determine connection durations
    if 'connection_durations' in config_analysis:
        logging.info('Determining connection durations')
        if multiprocess:
            connection_durations = con_ana.calc_connection_durations(graphs_cons, mp_pool=pool)
        else:
            connection_durations = con_ana.calc_connection_durations(graphs_cons)

        analysis_result['connection_durations'] = connection_durations[0]
        analysis_result['rehealing_times'] = connection_durations[1]
        connection_stats = con_ana.calc_connection_stats(
            connection_durations[0], graphs_cons[0].number_of_nodes())
        analysis_result['connection_duration_mean'] = connection_stats[0]
        analysis_result['connection_periods_mean'] = connection_stats[1]

    # Clean up and save results
    if multiprocess:
        pool.close()
        pool.join()

    utils.debug(time_start)

    utils.save(analysis_result, filepath_ana)
    return analysis_result
