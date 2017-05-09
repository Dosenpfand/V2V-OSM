""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# TODO: DEPRECATED, main script is now main_sim_osm.py!

# Standard imports
import time
import os
import multiprocessing as mp
from itertools import repeat
import pickle
import logging

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles
import propagation as prop
import sumo
import network_parser as nw_p
import connection_analysis as con_ana


def main_sim_single(network, max_pl=150):
    """Simulates the connections from one to all other vehicles using pathloss functions"""

    # Initialize
    vehs = network['vehs']
    graph_streets_wave = network['graph_streets_wave']
    gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    vehs.allocate(count_veh)

    # Find center vehicle
    time_start = utils.debug(None, 'Finding center vehicle')
    idx_center_veh = geom_o.find_center_veh(vehs.get())
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    utils.debug(time_start)

    # Determine propagation conditions
    time_start = utils.debug(None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos(vehs.get_points('center'),
                                     vehs.get_points('other'), gdf_buildings)
    vehs.add_key('nlos', idxs_other_vehs[is_nlos])
    is_olos_los = np.invert(is_nlos)
    vehs.add_key('olos_los', idxs_other_vehs[is_olos_los])
    utils.debug(time_start)

    # Determine OLOS and LOS
    time_start = utils.debug(None, 'Determining OLOS and LOS')
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos = prop.veh_cons_are_olos(vehs.get_points('center'),
                                     vehs.get_points('olos_los'), margin=2)
    is_los = np.invert(is_olos)
    vehs.add_key('olos', vehs.get_idxs('olos_los')[is_olos])
    vehs.add_key('los', vehs.get_idxs('olos_los')[is_los])
    utils.debug(time_start)

    # Determine orthogonal and parallel
    time_start = utils.debug(None, 'Determining orthogonal and parallel')

    is_orthogonal, coords_intersections = \
        prop.check_if_cons_orthogonal(graph_streets_wave,
                                      vehs.get_graph('center'),
                                      vehs.get_graph('nlos'),
                                      max_angle=np.pi)
    is_parallel = np.invert(is_orthogonal)
    vehs.add_key('orth', vehs.get_idxs('nlos')[is_orthogonal])
    vehs.add_key('par', vehs.get_idxs('nlos')[is_parallel])
    utils.debug(time_start)

    # Determining pathlosses for LOS and OLOS
    time_start = utils.debug(None, 'Calculating pathlosses for OLOS and LOS')

    p_loss = pathloss.Pathloss()
    distances_olos_los = np.sqrt(
        (vehs.get('olos_los')[:, 0] - vehs.get('center')[0])**2 +
        (vehs.get('olos_los')[:, 1] - vehs.get('center')[1])**2)

    pathlosses_olos = p_loss.pathloss_olos(distances_olos_los[is_olos])
    vehs.set_pathlosses('olos', pathlosses_olos)
    pathlosses_los = p_loss.pathloss_los(distances_olos_los[is_los])
    vehs.set_pathlosses('los', pathlosses_los)
    utils.debug(time_start)

    # Determining pathlosses for NLOS orthogonal
    time_start = utils.debug(
        None, 'Calculating pathlosses for NLOS orthogonal')

    # NOTE: Assumes center vehicle is receiver
    # NOTE: Uses airline vehicle -> intersection -> vehicle and not
    # street route
    distances_orth_tx = np.sqrt(
        (vehs.get('orth')[:, 0] - coords_intersections[is_orthogonal, 0])**2 +
        (vehs.get('orth')[:, 1] - coords_intersections[is_orthogonal, 1])**2)
    distances_orth_rx = np.sqrt(
        (vehs.get('center')[0] - coords_intersections[is_orthogonal, 0])**2 +
        (vehs.get('center')[1] - coords_intersections[is_orthogonal, 1])**2)
    pathlosses_orth = p_loss.pathloss_nlos(
        distances_orth_rx, distances_orth_tx)
    vehs.set_pathlosses('orth', pathlosses_orth)
    pathlosses_par = np.Infinity * np.ones(np.sum(is_parallel))
    vehs.set_pathlosses('par', pathlosses_par)
    utils.debug(time_start)

    # Determine in range / out of range
    time_start = utils.debug(None, 'Determining in range vehicles')
    idxs_in_range = vehs.get_pathlosses('other') < max_pl
    idxs_out_range = np.invert(idxs_in_range)
    vehs.add_key('in_range', vehs.get_idxs('other')[idxs_in_range])
    vehs.add_key('out_range', vehs.get_idxs('other')[idxs_out_range])
    utils.debug(time_start)


def main_sim_multiprocess_sumo(snapshot,
                               graph_streets,
                               gdf_buildings,
                               max_metric,
                               metric='distance'):
    """Runs a single snapshot analysis of a SUMO simulation result.
    Meant to be run in paralell"""

    # TODO: too much distance between SUMO vehicle positions and
    # OSMnx streets?
    vehs = sumo.vehicles_from_traces(
        graph_streets, snapshot)
    matrix_cons = con_ana.gen_connection_matrix(
        vehs,
        gdf_buildings,
        max_metric,
        metric=metric)

    return matrix_cons


def main_sim_multiprocess(iteration, densities_veh, static_params):
    """Runs the simulation using multiple processes"""

    np.random.seed(iteration)
    net_connectivities = np.zeros(np.size(densities_veh))
    path_redundancies = np.zeros(np.size(densities_veh), dtype=object)

    net = ox_a.load_network(static_params['place'],
                            which_result=static_params['which_result'])

    for idx_density, density in enumerate(densities_veh):
        logging.info('Started simulation with densitiy: {:.2E}, iteration: {:d}'.format(
            density, iteration))
        net_iter = net.copy()

        vehicles.place_vehicles_in_network(net_iter, density_veh=density,
                                           density_type=static_params['density_type'])
        graph_cons = \
            con_ana.gen_connection_graph(
                net_iter['vehs'],
                net_iter['gdf_buildings'],
                max_dist_olos_los=static_params['max_dist_olos_los'],
                max_dist_nlos=static_params['max_dist_nlos'])
        net_connectivity = con_ana.calc_net_connectivity(
            graph_cons, net_iter['vehs'])
        path_redundancy = con_ana.calc_path_redundancy(graph_cons, net['vehs'])
        net_connectivities[idx_density] = net_connectivity
        path_redundancies[idx_density] = path_redundancy

    return net_connectivities, path_redundancies


def main():
    """Main simulation function"""

    config = nw_p.params_from_conf()
    config_scenario = nw_p.params_from_conf(config['scenario'])
    config.update(config_scenario)

    if isinstance(config['densities_veh'], (list, tuple)):
        densities = np.zeros(0)
        for density_in in config['densities_veh']:
            if isinstance(density_in, dict):
                density = np.linspace(**density_in)
            else:
                density = density_in
            densities = np.append(densities, density)
        config['densities_veh'] = densities

    # Logger setup
    if 'loglevel' not in config:
        config['logelevel'] = 'ERROR'

    loglevel = logging.getLevelName(config['loglevel'])
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Setup OSMnx
    ox.config(log_console=True, log_level=loglevel, use_cache=True)

    # Switch to selected simulation mode
    if config['sim_mode'] == 'multi':
        net_connectivities = np.zeros(
            [config['iterations'], np.size(config['densities_veh'])])
        for iteration in np.arange(config['iterations']):
            for idx_density, density in enumerate(config['densities_veh']):
                logging.info('Started simulation with densitiy {:.2E}, iteration: {:d}'.format(
                    density, iteration))
                net = ox_a.load_network(config['place'],
                                        which_result=config['which_result'])
                vehicles.place_vehicles_in_network(net, density_veh=density,
                                                   density_type=config['density_type'])
                matrix_cons_snapshot = \
                    con_ana.gen_connection_graph(
                        net['vehs'],
                        net['gdf_buildings'],
                        max_dist_olos_los=config['max_dist_olos_los'],
                        max_dist_nlos=config['max_dist_nlos'])
                net_connectivity = con_ana.calc_net_connectivity(
                    matrix_cons_snapshot, net['vehs'])
                # TODO: !
                # path_redundancy = calc_path_redundancy(graph_cons, net['vehs'])

                net_connectivities[iteration, idx_density] = net_connectivity
            # TODO: Adapt filename and saved variable structure from
            # multiprocess!
            np.save('results/net_connectivities',
                    net_connectivities[:iteration + 1])

        if config['show_plot']:
            plot.plot_cluster_max(net['graph_streets'], net['gdf_buildings'],
                                  net['vehs'], show=False, place=config['place'])
            plt.show()

    elif config['sim_mode'] == 'multiprocess':
        time_start = time.time()

        # Prepare one network realization to download missing files
        ox_a.load_network(config['place'],
                          which_result=config['which_result'])

        with mp.Pool() as pool:
            sim_results = pool.starmap(main_sim_multiprocess, zip(
                range(config['iterations']), repeat(config['densities_veh']), repeat(config)))

        # Network connectivity results
        net_connectivities = np.zeros(
            [config['iterations'], np.size(config['densities_veh'])])
        for iter_index, iter_result in enumerate(sim_results):
            net_connectivities[iter_index, :] = iter_result[0]

        # Path redundancy results
        sim_results_arr = np.array(sim_results)
        path_redundancies = np.zeros(
            np.size(config['densities_veh']), dtype=object)
        for idx_density, density in enumerate(config['densities_veh']):
            path_redundancies[idx_density] = np.concatenate(
                sim_results_arr[:, 1, idx_density])

        # Save in and outputs
        time_finish = time.time()
        in_vars = config
        out_vars = {'net_connectivities': net_connectivities,
                    'path_redundancies': path_redundancies}
        info_vars = {'time_start': time_start, 'time_finish': time_finish}
        save_vars = {'in': in_vars, 'out': out_vars, 'info': info_vars}
        filepath_res = 'results/{:.0f}_{}.pickle'.format(
            time_finish, utils.string_to_filename(config['place']))
        with open(filepath_res, 'wb') as file:
            pickle.dump(save_vars, file)

        # Send mail
        if config['send_mail']:
            utils.send_mail_finish(config['mail_to'], time_start=time_start)

    elif config['sim_mode'] == 'single':
        if np.size(config['densities_veh']) > 1:
            raise ValueError(
                'Single simulation mode can only simulate 1 density value')

        net = ox_a.load_network(config['place'],
                                which_result=config['which_result'])
        vehicles.place_vehicles_in_network(net, density_veh=config['densities_veh'],
                                           density_type=config['density_type'])
        main_sim_single(net, max_pl=config['max_pl'])

        if config['show_plot']:
            plot.plot_prop_cond(net['graph_streets'], net['gdf_buildings'],
                                net['vehs'], show=False)
            plot.plot_pathloss(net['graph_streets'], net['gdf_buildings'],
                               net['vehs'], show=False)
            plot.plot_con_status(net['graph_streets'], net['gdf_buildings'],
                                 net['vehs'], show=False)
            plt.show()

    elif config['sim_mode'] == 'sumo':

        time_start = utils.debug(None, 'Loading street network')
        net = ox_a.load_network(config['place'],
                                which_result=config['which_result'])
        graph_streets = net['graph_streets']
        utils.debug(time_start)

        # Set unset SUMO settings to sane defaults
        if 'tls_settings' not in config['sumo']:
            config['sumo']['tls_settings'] = None
        if 'fringe_factor' not in config['sumo']:
            config['sumo']['fringe_factor'] = None
        if 'max_speed' not in config['sumo']:
            config['sumo']['max_speed'] = None
        if 'intermediate_points' not in config['sumo']:
            config['sumo']['intermediate_points'] = None
        if 'warmup_duration' not in config['sumo']:
            config['sumo']['warmup_duration'] = None
        if 'abort_after_sumo' not in config['sumo']:
            config['sumo']['abort_after_sumo'] = False
        if 'multiprocess' not in config:
            config['multiprocess'] = False

        for density_veh in config['densities_veh']:

            if config['density_type'] == 'absolute':
                count_veh = int(density_veh)
            elif config['density_type'] == 'length':
                street_lengths = geom_o.get_street_lengths(graph_streets)
                count_veh = int(round(density_veh * np.sum(street_lengths)))
            elif config['density_type'] == 'area':
                area = net['gdf_boundary'].area
                count_veh = int(round(density_veh * area))
            else:
                raise ValueError('Density type not supported')

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

            matrices_cons = np.zeros(veh_traces.size, dtype=object)

            if config['multiprocess']:
                with mp.Pool() as pool:
                    matrices_cons = pool.starmap(
                        main_sim_multiprocess_sumo,
                        zip(veh_traces,
                            repeat(net['graph_streets']),
                            repeat(net['gdf_buildings']),
                            repeat(config['max_connection_metric']),
                            repeat(config['connection_metric'])))
            else:
                for idx, snapshot in enumerate(veh_traces):
                    time_start = utils.debug(
                        None, 'Analyzing snapshot {:d}'.format(idx))
                    # TODO: too much distance between SUMO vehicle positions and
                    # OSMnx streets?
                    vehs = sumo.vehicles_from_traces(
                        net['graph_streets'], snapshot)
                    matrix_cons_snapshot = con_ana.gen_connection_matrix(
                        vehs,
                        net['gdf_buildings'],
                        max_metric=config['max_connection_metric'],
                        metric=config['connection_metric'])

                    matrices_cons[idx] = matrix_cons_snapshot

                    utils.debug(time_start)

            # Save in and outputs
            # TODO: save only current density!
            in_vars = config
            out_vars = {'matrices_cons': matrices_cons}
            # TODO: !
            # info_vars = {'time_start': time_start, 'time_finish': time_finish}
            # save_vars = {'in': in_vars, 'out': out_vars, 'info': info_vars}
            # TODO: maybe not save graphs but matrices to save space?
            save_vars = {'in': in_vars, 'out': out_vars}
            filepath_res = 'results/sumo_{}.{:d}.pickle'.format(
                utils.string_to_filename(config['place']), count_veh)
            utils.save(save_vars, filepath_res)

        # Send mail
        if config['send_mail']:
            # TODO: !
            # utils.send_mail_finish(config['mail_to'], time_start=time_start)
            utils.send_mail_finish(config['mail_to'])

        # TODO: modify?
        if config['show_plot']:
            time_start = utils.debug(None, 'Plotting animation')
            plot.plot_veh_traces_animation(
                veh_traces, net['graph_streets'], net['gdf_buildings'])
            utils.debug(time_start)

    else:
        raise NotImplementedError('Simulation type not supported')


if __name__ == '__main__':
    # Change to directory of script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Run main function
    main()
