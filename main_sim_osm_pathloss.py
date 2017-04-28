""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

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
import scipy.spatial.distance as dist
import networkx as nx
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


def main_sim_multi(network, max_dist_olos_los=250, max_dist_nlos=140):
    """Simulates all combinations of connections using only distances (not pathlosses) """

    # Initialize
    vehs = network['vehs']
    gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    max_dist = max(max_dist_nlos, max_dist_olos_los)
    count_cond = count_veh * (count_veh - 1) // 2
    vehs.allocate(count_cond)

    # Determine NLOS and OLOS/LOS
    time_start = utils.debug(None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos_all(
        vehs.get_points(), gdf_buildings, max_dist=max_dist)
    is_olos_los = np.invert(is_nlos)
    idxs_olos_los = np.where(is_olos_los)[0]
    idxs_nlos = np.where(is_nlos)[0]

    count_cond = count_veh * (count_veh - 1) // 2

    utils.debug(time_start)

    # Determine in range vehicles
    time_start = utils.debug(None, 'Determining in range vehicles')

    distances = dist.pdist(vehs.coordinates)
    idxs_in_range_olos_los = idxs_olos_los[
        distances[idxs_olos_los] < max_dist_olos_los]
    idxs_in_range_nlos = idxs_nlos[
        distances[idxs_nlos] < max_dist_nlos]
    idxs_in_range = np.append(
        idxs_in_range_olos_los, idxs_in_range_nlos)
    is_in_range = np.in1d(np.arange(count_cond), idxs_in_range)
    is_in_range_matrix = dist.squareform(is_in_range).astype(bool)
    # TODO: check if node names correspond to same indices as in vehs?
    graph_cons = nx.from_numpy_matrix(is_in_range_matrix)

    # Find biggest cluster
    clusters = nx.connected_component_subgraphs(graph_cons)
    cluster_max = max(clusters, key=len)
    net_connectivity = cluster_max.order() / count_veh
    vehs.add_key('cluster_max', cluster_max.nodes())
    not_cluster_max_nodes = np.arange(count_veh)[~np.in1d(
        np.arange(count_veh), cluster_max.nodes())]
    vehs.add_key('not_cluster_max', not_cluster_max_nodes)
    utils.debug(time_start)

    logging.info('Simulation result: Network connectivity {:.2f}%'.format(
        net_connectivity * 100))

    # Find center vehicle
    time_start = utils.debug(None, 'Finding center vehicle')
    idx_center_veh = geom_o.find_center_veh(vehs.get())
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    utils.debug(time_start)

    # Determine path redundancy
    # NOTE: we calculate the minimum number of node independent paths as an approximation (and not
    # the maximum)
    node_center_veh = idx_center_veh  # TODO: this does not seem to be the center?
    time_start = utils.debug(None, 'Determining path redundancy')

    path_redundancy = prop.path_redundancy(
        graph_cons, node_center_veh, distances)

    utils.debug(time_start)

    return net_connectivity, path_redundancy


def main_sim_single(network, max_pl=150):
    """Simulates the connections from one to all other vehicles using pathloss functions """

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
        net_connectivity, path_redundancy = \
            main_sim_multi(net_iter, max_dist_olos_los=static_params['max_dist_olos_los'],
                           max_dist_nlos=static_params['max_dist_nlos'])
        net_connectivities[idx_density] = net_connectivity
        path_redundancies[idx_density] = path_redundancy

    return net_connectivities, path_redundancies


def main():
    """Main simulation function"""

    sim_mode = 'sumo'  # 'single', 'multi', 'multiprocess', 'sumo'
    place = 'Landstrasse - Wien - Austria'
    use_pathloss = False  # TODO: Implement functions to use use_pathloss
    which_result = 1
    densities_veh = np.concatenate([np.arange(10, 90, 10), [120, 160]]) * 1e-6
    density_type = 'area'
    max_dist_olos_los = 250
    max_dist_nlos = 140
    iterations = 100
    max_pl = 150
    show_plot = False
    send_mail = True
    mail_to = 'markus.gasser@nt.tuwien.ac.at'
    loglevel = logging.DEBUG

    # Logger setup
    # TODO: loglevel does not propagate to modules!
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Adapt static input parameters
    static_params = {'place': place,
                     'which_result': which_result,
                     'density_type': density_type,
                     'max_dist_olos_los': max_dist_olos_los,
                     'max_dist_nlos': max_dist_nlos}

    # Setup OSMnx
    ox.config(log_console=True, log_level=loglevel, use_cache=True)

    # Switch to selected simulation mode
    if sim_mode == 'multi':
        net_connectivities = np.zeros([iterations, np.size(densities_veh)])
        for iteration in np.arange(iterations):
            for idx_density, density in enumerate(densities_veh):
                logging.info('Started simulation with densitiy {:.2E}, iteration: {:d}'.format(
                    density, iteration))
                net = ox_a.load_network(static_params['place'],
                                        which_result=static_params['which_result'])
                vehicles.place_vehicles_in_network(net, density_veh=density,
                                                   density_type=static_params['density_type'])
                net_connectivity, path_redundancy = \
                    main_sim_multi(net, max_dist_olos_los=max_dist_olos_los,
                                   max_dist_nlos=max_dist_nlos)
                net_connectivities[iteration, idx_density] = net_connectivity
            # TODO: Adapt filename and saved variable structure from
            # multiprocess!
            np.save('results/net_connectivities',
                    net_connectivities[:iteration + 1])

        if show_plot:
            plot.plot_cluster_max(net['graph_streets'], net['gdf_buildings'],
                                  net['vehs'], show=False, place=place)
            plt.show()

    elif sim_mode == 'multiprocess':
        time_start = time.time()

        # Prepare one network realization to download missing files
        ox_a.load_network(static_params['place'],
                          which_result=static_params['which_result'])

        with mp.Pool() as pool:
            sim_results = pool.starmap(main_sim_multiprocess, zip(
                range(iterations), repeat(densities_veh), repeat(static_params)))

        # Network connectivity results
        net_connectivities = np.zeros([iterations, np.size(densities_veh)])
        for iter_index, iter_result in enumerate(sim_results):
            net_connectivities[iter_index, :] = iter_result[0]

        # Path redundancy results
        sim_results_arr = np.array(sim_results)
        path_redundancies = np.zeros(np.size(densities_veh), dtype=object)
        for idx_density, density in enumerate(densities_veh):
            path_redundancies[idx_density] = np.concatenate(
                sim_results_arr[:, 1, idx_density])

        # Save in and outputs
        time_finish = time.time()
        in_vars = static_params
        in_vars['iterations'] = iterations
        in_vars['densities_veh'] = densities_veh
        out_vars = {'net_connectivities': net_connectivities,
                    'path_redundancies': path_redundancies}
        info_vars = {'time_start': time_start, 'time_finish': time_finish}
        save_vars = {'in': in_vars, 'out': out_vars, 'info': info_vars}
        filepath_res = 'results/{:.0f}_{}.pickle'.format(
            time_finish, utils.string_to_filename(place))
        with open(filepath_res, 'wb') as file:
            pickle.dump(save_vars, file)

        # Send mail
        if send_mail:
            utils.send_mail_finish(mail_to, time_start=time_start)

    elif sim_mode == 'single':
        if np.size(densities_veh) > 1:
            raise ValueError(
                'Single simulation mode can only simulate 1 density value')

        net = ox_a.load_network(static_params['place'],
                                which_result=static_params['which_result'])
        vehicles.place_vehicles_in_network(net, density_veh=densities_veh,
                                           density_type=static_params['density_type'])
        main_sim_single(net, max_pl=max_pl)

        if show_plot:
            plot.plot_prop_cond(net['graph_streets'], net['gdf_buildings'],
                                net['vehs'], show=False)
            plot.plot_pathloss(net['graph_streets'], net['gdf_buildings'],
                               net['vehs'], show=False)
            plot.plot_con_status(net['graph_streets'], net['gdf_buildings'],
                                 net['vehs'], show=False)
            plt.show()

    elif sim_mode == 'sumo':
        # TODO: expand!
        time_start = utils.debug(None, 'Loading street network')
        net = ox_a.load_network(static_params['place'],
                                which_result=static_params['which_result'])
        utils.debug(time_start)
        time_start = utils.debug(None, 'Loading vehicle traces')
        veh_traces = sumo.simple_wrapper(
            place, which_result=which_result, directory='sumo_data')
        utils.debug(time_start)

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
