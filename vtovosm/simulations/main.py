""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

import logging
import multiprocessing as mp
import os
import signal
import time
from itertools import repeat
from optparse import OptionParser

import numpy as np
from scipy.special import comb

from . import result_analysis
from .. import connection_analysis as con_ana
from .. import demo
from .. import geometry as geom_o
from .. import network_parser as nw_p
from .. import osmnx_addons as ox_a
from .. import plot
from .. import sumo
from .. import utils
from .. import vehicles

# Global variables
rte_count_con_checkpoint = 0
rte_count_con_total = 0
rte_time_start = 0
rte_time_checkpoint = 0


def parse_cmd_args():
    """Parses command line options"""

    parser = OptionParser()
    parser.add_option('-c', '--conf-file', dest='conf_path', default=None,
                      help='Load configuration from json FILE',
                      metavar='FILE')
    parser.add_option('-s', '--scenario', dest="scenario", default=None,
                      help='Use SCENARIO instead of the one defined in the configuration file',
                      metavar='SCENARIO')
    parser.add_option('-m', '--multi', action="store_true", dest="multi", default=False,
                      help="Simulate all scenarios defined in the configuration file")

    (options, args) = parser.parse_args()

    return options, args


def signal_handler(sig, frame):
    """Outputs simulation progress on SIGINFO"""

    if sig == signal.SIGTSTP:
        log_progress(rte_count_con_checkpoint, rte_count_con_total,
                     rte_time_checkpoint, rte_time_start)


def log_progress(c_checkpoint, c_end, t_checkpoint, t_start):
    """Estimates and logs the progress of the currently running simulation"""

    if c_checkpoint == 0:
        logging.info('No simulation progress and remaining time estimation possible')
        return

    t_now = time.time() - t_start
    c_now = c_checkpoint * t_now / t_checkpoint
    progress_now = min([c_now / c_end, 1])
    t_end = t_now * c_end / c_now
    t_todo = max([t_end - t_now, 0])
    logging.info(
        '{:.0f}% total simulation progress, '.format(progress_now * 100) +
        '{} remaining simulation time'.format(utils.seconds_to_string(t_todo)))


def sim_single_sumo(snapshot,
                    graph_streets,
                    gdf_buildings,
                    max_metric,
                    metric='distance',
                    graph_streets_wave=None):
    """Runs a single snapshot analysis of a SUMO simulation result.
    Can be run in parallel"""

    # TODO: too much distance between SUMO vehicle positions and
    # OSMnx streets?
    # Generate vehicles from SUMO traces snapshot
    vehs = sumo.vehicles_from_traces(
        graph_streets, snapshot)

    # Generate connection matrix
    matrix_cons = con_ana.gen_connection_matrix(
        vehs,
        gdf_buildings,
        max_metric,
        metric=metric,
        graph_streets_wave=graph_streets_wave)

    return matrix_cons, vehs


def sim_single_uniform(random_seed,
                       count_veh,
                       graph_streets,
                       gdf_buildings,
                       max_metric,
                       metric='distance',
                       graph_streets_wave=None):
    """Runs a single iteration of a simulation with uniform vehicle distribution.
    Can be run in parallel"""

    # Seed random number generator
    np.random.seed(random_seed)

    # Choose street indexes
    street_lengths = geom_o.get_street_lengths(graph_streets)
    rand_street_idxs = vehicles.choose_random_streets(
        street_lengths, count_veh)

    # Vehicle generation
    vehs = vehicles.generate_vehs(graph_streets, street_idxs=rand_street_idxs)

    # Generate connection matrix
    matrix_cons = con_ana.gen_connection_matrix(
        vehs,
        gdf_buildings,
        max_metric,
        metric=metric,
        graph_streets_wave=graph_streets_wave)

    return matrix_cons, vehs


def main_multi_scenario(conf_path=None, scenarios=None):
    """Simulates multiple scenarios"""

    # Load the configuration
    if scenarios is None:
        if conf_path is None:
            scenarios = nw_p.get_scenarios_list()
        else:
            scenarios = nw_p.get_scenarios_list(conf_path)

    if not isinstance(scenarios, (list, tuple)):
        raise RuntimeError('Single scenario not supported. Use appropriate function')

    # Iterate scenarios
    for scenario in scenarios:
        main(conf_path=conf_path, scenario=scenario)


def main(conf_path=None, scenario=None):
    """Main simulation function"""

    # TODO: why is global keyword needed?
    global rte_count_con_checkpoint
    global rte_count_con_total
    global rte_time_start
    global rte_time_checkpoint

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

    # Setup OSMnx
    # We are logging to dev/null as a workaround to get nice log output and so that specified levels are respected
    ox_a.setup()

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

    # Run time estimation
    if config['simulation_mode'] == 'demo':
        time_steps = 1
    elif config['distribution_veh'] == 'SUMO':
        time_steps = config['sumo']['sim_duration'] - \
                     config['sumo']['warmup_duration']
    elif config['distribution_veh'] == 'uniform':
        time_steps = config['iterations']

    rte_counts_con = comb(counts_veh, 2) * time_steps
    rte_count_con_total = np.sum(rte_counts_con)
    rte_time_start = time.time()
    rte_count_con_checkpoint = 0

    # Save start time
    time_start_total = time.time()

    # Iterate densities
    for idx_count_veh, count_veh in enumerate(counts_veh):

        # Determine results path and check if it exists
        if config['results_file_prefix'] is not None:
            filename_prefix = utils.string_to_filename(config['results_file_prefix'])
        elif 'scenario' in config:
            filename_prefix = utils.string_to_filename(config['scenario'])
        else:
            filename_prefix = utils.string_to_filename(config['place'])

        file_name = '{}.{:d}.pickle.xz'.format(filename_prefix, count_veh)

        if config['results_file_dir'] is not None:
            file_dir = config['results_file_dir']
        else:
            file_dir = 'results'

        filepath_res = os.path.join(file_dir, file_name)

        result_file_exists = os.path.isfile(filepath_res)
        if result_file_exists:
            if config['overwrite_result']:
                logging.warning('Results file already exists. Overwriting')
            else:
                logging.warning('Results file already exists. Skipping simulation')
                continue

        time_start_iter = time.time()
        logging.info('Simulating {:d} vehicles'.format(count_veh))

        if config['distribution_veh'] == 'SUMO':
            if not config['sumo']['skip_sumo']:
                # Run SUMO interface functions
                time_start = utils.debug(None, 'Running SUMO interface')
                veh_traces = sumo.simple_wrapper(
                    config['place'],
                    which_result=config['which_result'],
                    count_veh=count_veh,
                    duration=config['sumo']['sim_duration'],
                    warmup_duration=config['sumo']['warmup_duration'],
                    max_speed=config['sumo']['max_speed'],
                    tls_settings=config['sumo']['tls_settings'],
                    fringe_factor=config['sumo']['fringe_factor'],
                    intermediate_points=config['sumo']['intermediate_points'],
                    coordinate_tls=config['sumo']['coordinate_tls'],
                    directory=config['sumo']['directory'],
                    veh_rate_factor=config['sumo']['veh_rate_factor'])
                utils.debug(time_start)
            else:
                # Load vehicle traces
                time_start = utils.debug(None, 'Loading vehicle traces')
                veh_traces = sumo.load_veh_traces(
                    config['place'],
                    file_suffix=str(count_veh),
                    directory=config['sumo']['directory'],
                    delete_first_n=config['sumo']['warmup_duration'],
                    count_veh=count_veh)
                utils.debug(time_start)

            if config['sumo']['abort_after_sumo']:
                logger.warning('Aborting after SUMO completed')
                continue

        # Determine connected vehicles
        if config['simulation_mode'] == 'parallel':
            if config['distribution_veh'] == 'SUMO':
                if config['connection_metric'] == 'distance':
                    sim_param_list = \
                        zip(veh_traces,
                            repeat(net['graph_streets']),
                            repeat(net['gdf_buildings']),
                            repeat(config['max_connection_metric']),
                            repeat(config['connection_metric']))
                elif config['connection_metric'] == 'pathloss':
                    sim_param_list = \
                        zip(veh_traces,
                            repeat(net['graph_streets']),
                            repeat(net['gdf_buildings']),
                            repeat(config['max_connection_metric']),
                            repeat(config['connection_metric']),
                            repeat(net['graph_streets_wave']))
                else:
                    raise NotImplementedError(
                        'Connection metric not supported')
                with mp.Pool(processes=config['processes']) as pool:
                    mp_res = pool.starmap(
                        sim_single_sumo,
                        sim_param_list
                    )

            elif config['distribution_veh'] == 'uniform':
                random_seeds = np.arange(config['iterations'])

                if config['connection_metric'] == 'distance':
                    sim_param_list = \
                        zip(random_seeds,
                            repeat(count_veh),
                            repeat(net['graph_streets']),
                            repeat(net['gdf_buildings']),
                            repeat(config['max_connection_metric']),
                            repeat(config['connection_metric']))
                elif config['connection_metric'] == 'pathloss':
                    sim_param_list = \
                        zip(random_seeds,
                            repeat(count_veh),
                            repeat(net['graph_streets']),
                            repeat(net['gdf_buildings']),
                            repeat(config['max_connection_metric']),
                            repeat(config['connection_metric']),
                            repeat(net['graph_streets_wave']))
                else:
                    raise NotImplementedError(
                        'Connection metric not supported')
                with mp.Pool(processes=config['processes']) as pool:
                    mp_res = pool.starmap(
                        sim_single_uniform,
                        sim_param_list)

            else:
                raise NotImplementedError(
                    'Vehicle distribution type not supported')

            # Check result
            if len(mp_res) == 0:
                matrices_cons, vehs = [], []
            else:
                matrices_cons, vehs = list(zip(*mp_res))

            # Define which variables to save in a file
            results = {'matrices_cons': matrices_cons, 'vehs': vehs}

        elif config['simulation_mode'] == 'sequential':
            if config['distribution_veh'] == 'SUMO':
                matrices_cons = np.zeros(veh_traces.size, dtype=object)
                vehs = np.zeros(veh_traces.size, dtype=object)
                for idx, snapshot in enumerate(veh_traces):
                    time_start = utils.debug(
                        None, 'Analyzing snapshot {:d}'.format(idx))

                    if config['connection_metric'] == 'distance':
                        matrix_cons_snapshot, vehs_snapshot = \
                            sim_single_sumo(
                                snapshot,
                                net['graph_streets'],
                                net['gdf_buildings'],
                                max_metric=config['max_connection_metric'],
                                metric=config['connection_metric'])
                    elif config['connection_metric'] == 'pathloss':
                        matrix_cons_snapshot, vehs_snapshot = \
                            sim_single_sumo(
                                snapshot,
                                net['graph_streets'],
                                net['gdf_buildings'],
                                max_metric=config['max_connection_metric'],
                                metric=config['connection_metric'],
                                graph_streets_wave=net['graph_streets_wave'])
                    else:
                        raise NotImplementedError(
                            'Connection metric not supported')

                    matrices_cons[idx] = matrix_cons_snapshot
                    vehs[idx] = vehs_snapshot
                    utils.debug(time_start)
            elif config['distribution_veh'] == 'uniform':
                matrices_cons = np.zeros(config['iterations'], dtype=object)
                vehs = np.zeros(config['iterations'], dtype=object)
                for iteration in np.arange(config['iterations']):
                    time_start = utils.debug(
                        None, 'Analyzing iteration {:d}'.format(iteration))

                    if config['connection_metric'] == 'distance':
                        matrix_cons_snapshot, vehs_snapshot = \
                            sim_single_uniform(
                                iteration,
                                count_veh,
                                net['graph_streets'],
                                net['gdf_buildings'],
                                max_metric=config['max_connection_metric'],
                                metric=config['connection_metric'])
                    elif config['connection_metric'] == 'pathloss':
                        matrix_cons_snapshot, vehs_snapshot = \
                            sim_single_uniform(
                                iteration,
                                count_veh,
                                net['graph_streets'],
                                net['gdf_buildings'],
                                max_metric=config['max_connection_metric'],
                                metric=config['connection_metric'],
                                graph_streets_wave=net['graph_streets_wave'])
                    else:
                        raise NotImplementedError(
                            'Connection metric not supported')

                    matrices_cons[iteration] = matrix_cons_snapshot
                    vehs[iteration] = vehs_snapshot
                    utils.debug(time_start)
            else:
                raise NotImplementedError(
                    'Vehicle distribution type not supported')

            # Define which variables to save in a file
            results = {'matrices_cons': matrices_cons, 'vehs': vehs}

        elif config['simulation_mode'] == 'demo':
            vehicles.place_vehicles_in_network(net,
                                               density_veh=config['densities_veh'],
                                               density_type=config['density_type'])
            demo.simulate(net, max_pl=config['max_connection_metric'])

            # Define which variables to save in a file
            results = {'vehs': net['vehs']}

        else:
            raise NotImplementedError('Simulation mode not supported')

        # Progress report
        rte_time_checkpoint = time.time() - rte_time_start
        rte_count_con_checkpoint += rte_counts_con[idx_count_veh]
        log_progress(rte_count_con_checkpoint, rte_count_con_total,
                     rte_time_checkpoint, rte_time_start)

        # Save in and outputs
        config_save = config.copy()
        config_save['count_veh'] = count_veh

        time_finish_iter = time.time()
        info_vars = {'time_start': time_start_iter,
                     'time_finish': time_finish_iter}
        save_vars = {'config': config_save,
                     'results': results,
                     'info': info_vars}

        utils.save(save_vars, filepath_res)

    time_finish_total = time.time()
    runtime_total = time_finish_total - time_start_total
    logging.info('Total simulation runtime: {}'.format(utils.seconds_to_string(runtime_total)))

    # TODO: runtime estimation should also include result analysis!
    # Analyze simulation results
    if config['analyze_results'] is not None:
        if config['distribution_veh'] == 'SUMO' and config['sumo']['abort_after_sumo']:
            logging.warning('Not running result analysis because simulation was skipped')
        else:
            result_analysis.main(conf_path, scenario)

    # Send mail
    if config['send_mail']:
        utils.send_mail_finish(config['mail_to'], time_start=time_start_total)

    if config['save_plot']:
        if config['plot_dir'] is None:
            plot_dir = 'images'
        else:
            plot_dir = config['plot_dir']

        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        plot.setup()
        time_start = utils.debug(None, 'Plotting')

        if config['simulation_mode'] == 'demo':

            # Plot propagation conditions
            path = os.path.join(plot_dir, 'prop_cond.pdf')
            plot.plot_prop_cond(net['graph_streets'], net['gdf_buildings'],
                                net['vehs'], show=False, path=path, overwrite=config['overwrite_result'])

            # Plot pathloss
            path = os.path.join(plot_dir, 'pathloss.pdf')
            plot.plot_pathloss(net['graph_streets'], net['gdf_buildings'],
                               net['vehs'], show=False, path=path, overwrite=config['overwrite_result'])

            # Plot connection status
            path = os.path.join(plot_dir, 'con_status.pdf')
            plot.plot_con_status(net['graph_streets'], net['gdf_buildings'],
                                 net['vehs'], show=False, path=path, overwrite=config['overwrite_result'])
        elif config['distribution_veh'] == 'SUMO':

            if len(counts_veh) > 1:
                logging.warning('Multiple vehicle counts simulated, but will only generate plot for last one')

            # Plot animation of vehicle traces
            path = os.path.join(plot_dir, 'veh_traces.mp4')
            plot.plot_veh_traces_animation(
                veh_traces, net['graph_streets'], net['gdf_buildings'], show=False, path=path,
                overwrite=config['overwrite_result'])

            # Plot vehicle positions at the end of simulation time
            vehs_snapshot = veh_traces[-1]
            vehs = sumo.vehicles_from_traces(net['graph_streets'], vehs_snapshot)
            path = os.path.join(plot_dir, 'vehs_snapshot_end.pdf')
            plot.plot_vehs(net['graph_streets'], net['gdf_buildings'], vehs, show=False, path=path,
                           overwrite=config['overwrite_result'])

        utils.debug(time_start)


if __name__ == '__main__':
    # Parse command line options
    (options, _) = parse_cmd_args()

    # Register signal handler
    signal.signal(signal.SIGTSTP, signal_handler)

    # Run main simulation
    if options.multi:
        main_multi_scenario(conf_path=options.conf_path, scenarios=options.scenario)
    else:
        main(conf_path=options.conf_path, scenario=options.scenario)
