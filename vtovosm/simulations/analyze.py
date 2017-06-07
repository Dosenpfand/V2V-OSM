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
    for idx_count_veh, count_veh in enumerate(counts_veh):

        # Determine results path and check if it exists
        if config['results_file_prefix'] is not None:
            filename_prefix = utils.string_to_filename(config['results_file_prefix'])
        elif 'scenario' in config:
            filename_prefix = utils.string_to_filename(config['scenario'])
        else:
            filename_prefix = utils.string_to_filename(config['place'])

        file_name_res = '{}.{:d}.pickle.gz'.format(filename_prefix, count_veh)
        file_name_ana = '{}.{:d}_analysis.pickle.gz'.format(filename_prefix, count_veh)

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

    logging.info('Starting analyis of results')
    if config['simulation_mode'] == 'parallel':
        with mp.Pool() as pool:
            param_list = zip(filepaths_res,
                             filepaths_ana,
                             repeat(config['analyze_results']))

            pool.starmap(analyze_single, param_list)

    elif config['simulation_mode'] == 'sequential':
        for filepath_res, filepath_ana in zip(filepaths_res, filepaths_ana):
            analyze_single(filepath_res, filepath_ana, config['analyze_results'])

    else:
        raise NotImplementedError('Mode not supported')

def analyze_single(filepath_res, filepath_ana, config_analysis):
    """Runs a single vehicle count analysis of a simulation result.
    Can be run in parallel"""

    results_loaded = utils.load(filepath_res)
    matrices_cons = results_loaded['results']['matrices_cons']
    vehs = results_loaded['results']['vehs']

    # Analyze results
    if config_analysis is not None:
        time_start = utils.debug(None, 'Analyzing results')

        if config_analysis == ['all']:
            config_analysis = ['net_connectivities',
                               'path_redundancies',
                               'link_durations',
                               'connection_durations']

        graphs_cons = []
        for matrix_cons in matrices_cons:
            graphs_cons.append(nx.from_numpy_matrix(matrix_cons))

        analysis_result = {}
        for analysis in config_analysis:
            if analysis == 'net_connectivities':
                logging.info('Determining network connectivities')
                net_connectivities = con_ana.calc_net_connectivities(graphs_cons)
                analysis_result['net_connectivities'] = net_connectivities
            elif analysis == 'path_redundancies':
                logging.info('Determining path redundancies')
                path_redundancies = con_ana.calc_center_path_redundancies(graphs_cons, vehs)
                analysis_result['path_redundancies'] = path_redundancies
            elif analysis == 'link_durations':
                logging.info('Determining link durations')
                link_durations = con_ana.calc_link_durations(graphs_cons)
                analysis_result['link_durations'] = link_durations
            elif analysis == 'connection_durations':
                logging.info('Determining connection durations')
                connection_durations = con_ana.calc_connection_durations(graphs_cons)
                analysis_result['connection_durations'] = connection_durations[0]
                analysis_result['rehealing_times'] = connection_durations[1]
                connection_stats = con_ana.calc_connection_stats(
                    connection_durations[0], graphs_cons[0].number_of_nodes())
                analysis_result['connection_duration_mean'] = connection_stats[0]
                analysis_result['connection_periods_mean'] = connection_stats[1]
            else:
                raise NotImplementedError('Analysis not supported')

        utils.debug(time_start)

        utils.save(analysis_result, filepath_ana)
        return analysis_result
