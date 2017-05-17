""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import time
import os
import multiprocessing as mp
from itertools import repeat
import logging
import datetime
import signal

# Extension imports
import numpy as np
import osmnx as ox
from scipy.special import comb

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles
import sumo
import network_parser as nw_p
import connection_analysis as con_ana

# Global variables
rte_count_con_checkpoint = 0
rte_count_con_total = 0
rte_time_start = 0
rte_time_checkpoint = 0


def signal_handler(sig, frame):
    """Outputs simulation progress on SIGINFO"""

    if sig == signal.SIGTSTP:
        log_progress(rte_count_con_checkpoint, rte_count_con_total,
                     rte_time_checkpoint, rte_time_start)


def log_progress(c_checkpoint, c_end, t_checkpoint, t_start):
    """Estimates and logs the progress of the currently running simulation"""

    if c_checkpoint == 0:
        logging.info('No progress and remaining time estimation possible')
        return

    t_now = time.time() - t_start
    c_now = c_checkpoint * t_now / t_checkpoint
    progress_now = min([c_now / c_end, 1])
    t_end = t_now * c_end / c_now
    t_todo = max([t_end - t_now, 0])

    datetime_todo = datetime.datetime(
        1, 1, 1) + datetime.timedelta(seconds=int(t_todo))
    logging.info(
        '{:.0f}% total simulation progress, '.format(progress_now * 100) +
        '{:02d}:{:02d}:{:02d}:{:02d} remaining simulation time'.format(
            datetime_todo.day - 1,
            datetime_todo.hour,
            datetime_todo.minute,
            datetime_todo.second))


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

    return matrix_cons


def sim_single_uniform(random_seed,
                       count_veh,
                       graph_streets,
                       gdf_buildings,
                       max_metric,
                       metric='distance',
                       graph_streets_wave=None):
    """Runs a single iteration of a simulation with uniform vehicle distribution.
    Can be run in paralell"""

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

    return matrix_cons


def main():
    """Main simulation function"""

    # TODO: why is global keyword needed?
    global rte_count_con_checkpoint
    global rte_count_con_total
    global rte_time_start
    global rte_time_checkpoint

    # General setup
    time_start_total = time.time()
    config = nw_p.params_from_conf()
    config_scenario = nw_p.params_from_conf(config['scenario'])
    config.update(config_scenario)
    densities_veh = utils.convert_config_densities(config['densities_veh'])

    # Logger setup
    if 'loglevel' not in config:
        config['logelevel'] = 'ERROR'

    loglevel = logging.getLevelName(config['loglevel'])
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Setup OSMnx
    ox.config(log_console=True, log_level=loglevel, use_cache=True)

    # Load street network
    time_start = utils.debug(None, 'Loading street network')
    net = ox_a.load_network(config['place'],
                            which_result=config['which_result'])
    graph_streets = net['graph_streets']
    utils.debug(time_start)

    # Sanitize config
    config = utils.fill_config(config)

    # Convert vehicle densities to counts
    # Iterate densities
    counts_veh = np.zeros(densities_veh.size, dtype=int)

    if config['density_type'] == 'length':
        street_lengths = geom_o.get_street_lengths(graph_streets)

    for idx, density_veh in enumerate(densities_veh):
        # Determine total vehicle count
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
    if config['distribution_veh'] == 'SUMO':
        time_steps = config['sumo']['sim_duration'] - \
            config['sumo']['warmup_duration']
    elif config['distribution_veh'] == 'uniform':
        time_steps = config['iterations']

    rte_counts_con = comb(counts_veh, 2) * time_steps
    rte_count_con_total = np.sum(rte_counts_con)
    rte_time_start = time.time()
    rte_count_con_checkpoint = 0

    # Iterate densities
    for idx_count_veh, count_veh in enumerate(counts_veh):
        logging.info('Simulating {:d} vehicles'.format(count_veh))

        if config['distribution_veh'] == 'SUMO':
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
                directory='sumo_data')
            utils.debug(time_start)

            if config['sumo']['abort_after_sumo']:
                continue

        # Determine connected vehicles
        if config['simulation_mode'] == 'parallel':
            with mp.Pool() as pool:
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

                    matrices_cons = pool.starmap(
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

                    matrices_cons = pool.starmap(
                        sim_single_uniform,
                        sim_param_list)
                else:
                    raise NotImplementedError(
                        'Vehicle distribution type not supported')
        elif config['simulation_mode'] == 'sequential':
            if config['distribution_veh'] == 'SUMO':
                matrices_cons = np.zeros(veh_traces.size, dtype=object)
                for idx, snapshot in enumerate(veh_traces):
                    time_start = utils.debug(
                        None, 'Analyzing snapshot {:d}'.format(idx))

                    if config['connection_metric'] == 'distance':
                        matrix_cons_snapshot = \
                            sim_single_sumo(
                                snapshot,
                                net['graph_streets'],
                                net['gdf_buildings'],
                                max_metric=config['max_connection_metric'],
                                metric=config['connection_metric'])
                    elif config['connection_metric'] == 'pathloss':
                        matrix_cons_snapshot = \
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
                    utils.debug(time_start)
            elif config['distribution_veh'] == 'uniform':
                matrices_cons = np.zeros(config['iterations'], dtype=object)
                for iteration in np.arange(config['iterations']):
                    time_start = utils.debug(
                        None, 'Analyzing iteration {:d}'.format(iteration))

                    if config['connection_metric'] == 'distance':
                        matrix_cons_snapshot = \
                            sim_single_uniform(
                                iteration,
                                count_veh,
                                net['graph_streets'],
                                net['gdf_buildings'],
                                max_metric=config['max_connection_metric'],
                                metric=config['connection_metric'])
                    elif config['connection_metric'] == 'pathloss':
                        matrix_cons_snapshot = \
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
                    utils.debug(time_start)
            else:
                raise NotImplementedError(
                    'Vehicle distribution type not supported')

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
        results = {'matrices_cons': matrices_cons}
        time_finish_total = time.time()
        info_vars = {'time_start': time_start_total,
                     'time_finish': time_finish_total}
        save_vars = {'config': config_save,
                     'results': results,
                     'info': info_vars}
        filepath_res = 'results/{}.{:d}.pickle.gz'.format(
            utils.string_to_filename(config['place']), count_veh)
        utils.save(save_vars, filepath_res)

    # Send mail
    if config['send_mail']:
        utils.send_mail_finish(config['mail_to'], time_start=time_start_total)

    # TODO: modify?
    if config['show_plot']:
        time_start = utils.debug(None, 'Plotting animation')
        plot.plot_veh_traces_animation(
            veh_traces, net['graph_streets'], net['gdf_buildings'])
        utils.debug(time_start)


if __name__ == '__main__':

    # Register signal handler
    signal.signal(signal.SIGTSTP, signal_handler)

    # Change to directory of script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run main function
    main()
