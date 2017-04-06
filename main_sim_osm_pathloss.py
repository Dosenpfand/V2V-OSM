""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import time
import os.path
import multiprocessing as mp
from itertools import repeat
import pickle

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import scipy.stats as st
import networkx as nx

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles
import propagation as prop


def prepare_network(place, which_result=1, density_veh=100, density_type='absolute', debug=False):
    """Generates streets, buildings and vehicles """

    # Setup
    ox_a.setup(debug)

    # Load data
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(
        utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(
        utils.string_to_filename(place))
    filename_data_boundary = 'data/{}_boundary.pickle'.format(
        utils.string_to_filename(place))
    filename_data_wave = 'data/{}_wave.pickle'.format(
        utils.string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings) and \
            os.path.isfile(filename_data_boundary):
        # Load from file
        time_start = utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        time_start = utils.debug(debug, None, 'Loading data from the internet')
        data = ox_a.download_place(place, which_result=which_result)

    graph_streets = data['streets']
    gdf_buildings = data['buildings']
    ox_a.add_geometry(graph_streets)

    utils.debug(debug, time_start)

    # Generate wave propagation graph:
    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    if os.path.isfile(filename_data_wave):
        # Load from file
        time_start = utils.debug(
            debug, None, 'Loading graph for wave propagation')
        with open(filename_data_wave, 'rb') as file:
            graph_streets_wave = pickle.load(file)
    else:
        # Generate
        time_start = utils.debug(
            debug, None, 'Generating graph for wave propagation')
        graph_streets_wave = graph_streets.to_undirected()
        # TODO: check if add_edges_if_los() is really working!!!
        prop.add_edges_if_los(graph_streets_wave, gdf_buildings)
        with open(filename_data_wave, 'wb') as file:
            pickle.dump(graph_streets_wave, file)

    utils.debug(debug, time_start)

    # Streets and positions selection
    time_start = utils.debug(debug, None, 'Choosing random vehicle positions')

    street_lengths = geom_o.get_street_lengths(graph_streets)

    if density_type == 'absolute':
        count_veh = int(density_veh)
    elif density_type == 'length':
        count_veh = int(round(density_veh * np.sum(street_lengths)))
    elif density_type == 'area':
        area = data['boundary'].area
        count_veh = int(round(density_veh * area))
    else:
        raise ValueError('Density type not supported')

    rand_street_idxs = vehicles.choose_random_streets(
        street_lengths, count_veh)
    utils.debug(debug, time_start)

    # Vehicle generation
    time_start = utils.debug(debug, None, 'Generating vehicles')
    vehs = vehicles.generate_vehs(graph_streets, rand_street_idxs)
    utils.debug(debug, time_start)

    network = {'graph_streets': graph_streets, 'graph_streets_wave': graph_streets_wave,
               'gdf_buildings': gdf_buildings, 'vehs': vehs}

    return network


def main_sim_multi(network, max_dist_olos_los=250, max_dist_nlos=140, debug=False):
    """Simulates all combinations of connections using only distances (not pathlosses) """

    # Initialize
    vehs = network['vehs']
    gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    max_dist = max(max_dist_nlos, max_dist_olos_los)
    count_cond = count_veh * (count_veh - 1) // 2
    vehs.allocate(count_cond)

    # Determine NLOS and OLOS/LOS
    time_start = utils.debug(
        debug, None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos_all(
        vehs.get_points(), gdf_buildings, max_dist=max_dist)
    is_olos_los = np.invert(is_nlos)
    idxs_olos_los = np.where(is_olos_los)[0]
    idxs_nlos = np.where(is_nlos)[0]

    count_cond = count_veh * (count_veh - 1) // 2

    utils.debug(debug, time_start)

    # Determine in range vehicles
    time_start = utils.debug(
        debug, None, 'Determining in range vehicles')

    distances = dist.pdist(vehs.coordinates)
    idxs_in_range_olos_los = idxs_olos_los[
        distances[idxs_olos_los] < max_dist_olos_los]
    idxs_in_range_nlos = idxs_nlos[
        distances[idxs_nlos] < max_dist_nlos]
    idxs_in_range = np.append(
        idxs_in_range_olos_los, idxs_in_range_nlos)
    is_in_range = np.in1d(np.arange(count_cond), idxs_in_range)
    is_in_range_matrix = dist.squareform(is_in_range).astype(bool)
    graph_cons = nx.from_numpy_matrix(is_in_range_matrix)

    # Find biggest cluster
    clusters = nx.connected_component_subgraphs(graph_cons)
    cluster_max = max(clusters, key=len)
    net_connectivity = cluster_max.order() / count_veh
    vehs.add_key('cluster_max', cluster_max.nodes())
    not_cluster_max_nodes = np.arange(count_veh)[~np.in1d(
        np.arange(count_veh), cluster_max.nodes())]
    vehs.add_key('not_cluster_max', not_cluster_max_nodes)
    utils.debug(debug, time_start)
    if debug:
        print('Network connectivity: {:.2f} %'.format(
            net_connectivity * 100))

    return net_connectivity


def main_sim(network, max_pl=150, debug=False):
    """Simulates the connections from one to all other vehicles using pathloss functions """

    # Initialize
    vehs = network['vehs']
    graph_streets_wave = network['graph_streets_wave']
    gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    vehs.allocate(count_veh)

    # Find center vehicle
    time_start = utils.debug(debug, None, 'Finding center vehicle')
    idx_center_veh = geom_o.find_center_veh(vehs.get())
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)

    # Determine propagation conditions
    time_start = utils.debug(
        debug, None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos(vehs.get_points('center'),
                                     vehs.get_points('other'), gdf_buildings)
    vehs.add_key('nlos', idxs_other_vehs[is_nlos])
    is_olos_los = np.invert(is_nlos)
    vehs.add_key('olos_los', idxs_other_vehs[is_olos_los])
    utils.debug(debug, time_start)

    # Determine OLOS and LOS
    time_start = utils.debug(debug, None, 'Determining OLOS and LOS')
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos = prop.veh_cons_are_olos(vehs.get_points('center'),
                                     vehs.get_points('olos_los'), margin=2)
    is_los = np.invert(is_olos)
    vehs.add_key('olos', vehs.get_idxs('olos_los')[is_olos])
    vehs.add_key('los', vehs.get_idxs('olos_los')[is_los])
    utils.debug(debug, time_start)

    # Determine orthogonal and parallel
    time_start = utils.debug(
        debug, None, 'Determining orthogonal and parallel')

    is_orthogonal, coords_intersections = \
        prop.check_if_cons_orthogonal(graph_streets_wave,
                                      vehs.get_graph('center'),
                                      vehs.get_graph('nlos'),
                                      max_angle=np.pi)
    is_parallel = np.invert(is_orthogonal)
    vehs.add_key('orth', vehs.get_idxs('nlos')[is_orthogonal])
    vehs.add_key('par', vehs.get_idxs('nlos')[is_parallel])
    utils.debug(debug, time_start)

    # Determining pathlosses for LOS and OLOS
    time_start = utils.debug(
        debug, None, 'Calculating pathlosses for OLOS and LOS')

    p_loss = pathloss.Pathloss()
    distances_olos_los = np.sqrt(
        (vehs.get('olos_los')[:, 0] - vehs.get('center')[0])**2 +
        (vehs.get('olos_los')[:, 1] - vehs.get('center')[1])**2)

    # TODO: why - ? fix in pathloss.py
    pathlosses_olos = - \
        p_loss.pathloss_olos(distances_olos_los[is_olos])
    vehs.set_pathlosses('olos', pathlosses_olos)
    # TODO: why - ? fix in pathloss.py
    pathlosses_los = -p_loss.pathloss_los(distances_olos_los[is_los])
    vehs.set_pathlosses('los', pathlosses_los)
    utils.debug(debug, time_start)

    # Determining pathlosses for NLOS orthogonal
    time_start = utils.debug(
        debug, None, 'Calculating pathlosses for NLOS orthogonal')

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
    utils.debug(debug, time_start)

    # Determine in range / out of range
    time_start = utils.debug(
        debug, None, 'Determining in range vehicles')
    idxs_in_range = vehs.get_pathlosses('other') < max_pl
    idxs_out_range = np.invert(idxs_in_range)
    vehs.add_key('in_range', vehs.get_idxs('other')[idxs_in_range])
    vehs.add_key('out_range', vehs.get_idxs('other')[idxs_out_range])
    utils.debug(debug, time_start)


def multiprocess_sim(iteration, densities_veh, static_params):

    np.random.seed(iteration)
    net_connectivities = np.zeros(np.size(densities_veh))

    for idx_density, density in enumerate(densities_veh):
        print('Densitiy: {:.2E}, Iteration: {:d}'.format(
            density, iteration))
        net = prepare_network(static_params['place'], which_result=static_params['which_result'],
                              density_veh=density, density_type=static_params[
                                  'density_type'],
                              debug=False)
        net_connectivity = main_sim_multi(net, max_dist_olos_los=static_params['max_dist_olos_los'],
                                          max_dist_nlos=static_params['max_dist_nlos'], debug=False)
        net_connectivities[idx_density] = net_connectivity

    return net_connectivities


def net_connectivity_stats(net_connectivities, confidence=0.95):
    # TODO: !
    mean = np.mean(net_connectivities)
    conf_intervals = st.t.interval(confidence, len(
        net_connectivities) - 1, loc=mean, scale=st.sem(net_connectivities))

    return mean, conf_intervals


def plot_net_connectivity():
    # TODO: plot confidence intervals
    net_connectivities = np.load('results/net_connectivities.npy')
    aver_net_cons = np.mean(net_connectivities, axis=0) * 100
    aver_net_cons_paper = np.array([12.67, 18.92, 21.33,
                                    34.75, 69.72, 90.05, 97.46, 98.97, 99.84, 100])
    conf_net_cons_paper = np.array([2.22, 4.51, 2.57,
                                    6.58, 8.02, 3.48, 1.25, 0.61, 0.25, 0]) / 100 * aver_net_cons_paper
    net_densities = np.concatenate([np.arange(10, 90, 10), [120, 160]])

    plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    plt.rc('text', usetex=True)
    plt.plot(net_densities, aver_net_cons, label='OSM (own method)')
    plt.errorbar(net_densities, aver_net_cons_paper, conf_net_cons_paper,
                 label='Manhattan grid (Viriyasitavat et al.)')

    # Add additional information to plot

    plt.xlabel(r'Network density $[veh/km^2]$')
    plt.ylabel(r'Average network connectivity [\%]')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # TODO: what happens when multiprocess simulation is started and data files are not present?
    # TODO: argparse!
    sim_mode = 'multiprocess'  # 'single', 'multi', 'multiprocess'
    place = 'Upper West Side - New York - USA'
    # TODO: Implement functions to use use_pathloss
    use_pathloss = False
    which_result = 1
    densities_veh = np.concatenate([np.arange(10, 90, 10), [120, 160]]) * 1e-6
    density_type = 'area'
    max_dist_olos_los = 250
    max_dist_nlos = 140
    iterations = 100
    max_pl = 150
    show_plot = False

    if sim_mode == 'multi':
        net_connectivities = np.zeros([iterations, np.size(densities_veh)])
        for iteration in np.arange(iterations):
            for idx_density, density in enumerate(densities_veh):
                print('Densitiy: {:.2E}, Iteration: {:d}'.format(
                    density, iteration))
                net = prepare_network(place, which_result=which_result, density_veh=density,
                                      density_type=density_type, debug=True)
                net_connectivity = main_sim_multi(net, max_dist_olos_los=max_dist_olos_los,
                                                  max_dist_nlos=max_dist_nlos, debug=True)
                net_connectivities[iteration, idx_density] = net_connectivity
            np.save('results/net_connectivities',
                    net_connectivities[:iteration + 1])

        if show_plot:
            plot.plot_cluster_max(net['graph_streets'], net['gdf_buildings'],
                                  net['vehs'], show=False, place=place)
            plt.show()

    elif sim_mode == 'multiprocess':
        static_params = {'place': place,
                         'which_result': which_result,
                         'density_type': density_type,
                         'max_dist_olos_los': max_dist_olos_los,
                         'max_dist_nlos': max_dist_nlos}

        # Prepare one network realization to download missing files
        # TODO: seperate download function
        prepare_network(place, which_result=which_result,
                        density_veh=0, density_type='absolute',
                        debug=False)

        net_connectivities = np.zeros([iterations, np.size(densities_veh)])
        with mp.Pool() as pool:
            net_connectivities = pool.starmap(multiprocess_sim, zip(
                range(iterations), repeat(densities_veh), repeat(static_params)))

        # Save in and outputs
        in_vars = static_params
        in_vars['iterations'] = iterations
        in_vars['densities_veh'] = densities_veh
        out_vars = {'net_connectivities': net_connectivities}
        save_vars = {'in': in_vars, 'out': out_vars}
        finish_time = time.time()
        filepath_res = 'results/{:.0f}.pickle'.format(finish_time)
        with open(filepath_res, 'wb') as file:
            pickle.dump(save_vars, file)

    elif sim_mode == 'single':
        net = prepare_network(place, which_result=which_result, density_veh=densities_veh,
                              density_type=density_type, debug=True)
        main_sim(net, max_pl=max_pl, debug=True)

        if show_plot:
            plot.plot_prop_cond(net['graph_streets'], net['gdf_buildings'],
                                net['vehs'], show=False)
            plot.plot_pathloss(net['graph_streets'], net['gdf_buildings'],
                               net['vehs'], show=False)
            plot.plot_con_status(net['graph_streets'], net['gdf_buildings'],
                                 net['vehs'], show=False)
            plt.show()

    else:
        raise NotImplementedError('Simulation type not supported')
