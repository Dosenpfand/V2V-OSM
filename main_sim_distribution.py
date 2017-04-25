""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import time
import os.path
import multiprocessing as mp
from itertools import repeat
import pickle
from itertools import repeat

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import networkx as nx

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles as vehicle_manager
import propagation as prop
import shapely.geometry as geo


def load_network(place, which_result=1, debug=False):
    """Generates streets and buildings and"""

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
        # time_start = # utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        # time_start = # utils.debug(debug, None, 'Loading data from the
        # internet')
        data = ox_a.download_place(place, which_result=which_result)

    graph_streets = data['streets']
    gdf_buildings = data['buildings']
    gdf_boundary = data['boundary']
    ox_a.add_geometry(graph_streets)

    # utils.debug(debug, # time_start)

    # Generate wave propagation graph:
    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    if os.path.isfile(filename_data_wave):
        # Load from file
        # time_start = # utils.debug(
        #    debug, None, 'Loading graph for wave propagation')
        with open(filename_data_wave, 'rb') as file:
            graph_streets_wave = pickle.load(file)
    else:
        # Generate
        # time_start = # utils.debug(
        #    debug, None, 'Generating graph for wave propagation')
        graph_streets_wave = graph_streets.to_undirected()
        # TODO: check if add_edges_if_los() is really working!!!
        prop.add_edges_if_los(graph_streets_wave, gdf_buildings)
        with open(filename_data_wave, 'wb') as file:
            pickle.dump(graph_streets_wave, file)

    # utils.debug(debug, # time_start)

    network = {'graph_streets': graph_streets,
               'graph_streets_wave': graph_streets_wave,
               'gdf_buildings': gdf_buildings,
               'gdf_boundary': gdf_boundary}

    return network


def generate_vehicles(network, density_veh=100,
                      density_type='absolute', debug=False):
    """Generates vehicles in the network"""

    graph_streets = network['graph_streets']

    # Streets and positions selection
    # time_start = # utils.debug(debug, None, 'Choosing random vehicle
    # positions')

    street_lengths = geom_o.get_street_lengths(graph_streets)

    if density_type == 'absolute':
        count_veh = int(density_veh)
    elif density_type == 'length':
        count_veh = int(round(density_veh * np.sum(street_lengths)))
    elif density_type == 'area':
        area = network['gdf_boundary'].area
        count_veh = int(round(density_veh * area))
    else:
        raise ValueError('Density type not supported')

    rand_street_idxs = vehicle_manager.choose_random_streets(
        street_lengths, count_veh)
    # utils.debug(debug, # time_start)

    # Vehicle generation
    # time_start = # utils.debug(debug, None, 'Generating vehicles')
    vehs = vehicle_manager.generate_vehs(graph_streets, rand_street_idxs)
    # utils.debug(debug, # time_start)

    network['vehs'] = vehs
    return vehs


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
    # time_start = # utils.debug(
    #    debug, None, 'Determining propagation conditions')
    is_nlos=prop.veh_cons_are_nlos_all(
        vehs.get_points(), gdf_buildings, max_dist = max_dist)
    is_olos_los=np.invert(is_nlos)
    idxs_olos_los=np.where(is_olos_los)[0]
    idxs_nlos=np.where(is_nlos)[0]

    count_cond=count_veh * (count_veh - 1) // 2

    # utils.debug(debug, # time_start)

    # Determine in range vehicles
    # time_start = # utils.debug(
    #    debug, None, 'Determining in range vehicles')

    distances=dist.pdist(vehs.coordinates)
    idxs_in_range_olos_los=idxs_olos_los[
        distances[idxs_olos_los] < max_dist_olos_los]
    idxs_in_range_nlos=idxs_nlos[
        distances[idxs_nlos] < max_dist_nlos]
    idxs_in_range=np.append(
        idxs_in_range_olos_los, idxs_in_range_nlos)
    is_in_range=np.in1d(np.arange(count_cond), idxs_in_range)
    is_in_range_matrix=dist.squareform(is_in_range).astype(bool)
    # TODO: check if node names correspond to same indices as in vehs?
    graph_cons=nx.from_numpy_matrix(is_in_range_matrix)

    # Find biggest cluster
    clusters=nx.connected_component_subgraphs(graph_cons)
    cluster_max=max(clusters, key=len)
    net_connectivity=cluster_max.order() / count_veh
    vehs.add_key('cluster_max', cluster_max.nodes())
    not_cluster_max_nodes=np.arange(count_veh)[~np.in1d(
        np.arange(count_veh), cluster_max.nodes())]
    vehs.add_key('not_cluster_max', not_cluster_max_nodes)
    # utils.debug(debug, # time_start)

    if debug:
        print('Network connectivity: {:.2f} %'.format(
            net_connectivity * 100))

    # Find center vehicle
    # time_start = # utils.debug(debug, None, 'Finding center vehicle')
    idx_center_veh=geom_o.find_center_veh(vehs.get())
    idxs_other_vehs=np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    # utils.debug(debug, # time_start)

    # Determine path redundancy
    # NOTE: we calculate the minimum number of node independent paths as an approximation (and not
    # the maximum)
    node_center_veh=idx_center_veh  # TODO: this does not seem to be the center?
    # time_start = # utils.debug(debug, None, 'Determining path redundancy')

    path_redundancy=prop.path_redundancy(
        graph_cons, node_center_veh, distances)

    # utils.debug(debug, # time_start)

    return net_connectivity, path_redundancy


def main_sim(network, max_pl=150, debug=False, max_dist_olos_los=300,center_veh=0):
    """Simulates the connections from one to all other vehicles using pathloss functions """
    idx = center_veh
    # Initialize
    vehs=network['vehs']
    graph_streets_wave=network['graph_streets_wave']
    gdf_buildings=network['gdf_buildings']
    count_veh=vehs.count
    vehs.allocate(count_veh)
    # Find center vehicle
    # time_start = # utils.debug(debug, None, 'Finding center vehicle')

    print("\t Current Run: {}/{}".format(idx,count_veh))
    idx_center_veh=center_veh
    idxs_other_vehs=np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    # utils.debug(debug, # time_start)

    # Determine propagation conditions
    # time_start = # utils.debug(
    #    debug, None, 'Determining propagation conditions')
    is_nlos=prop.veh_cons_are_nlos(vehs.get_points('center'),
                                        vehs.get_points('other'),
                                        gdf_buildings,
                                        max_dist=max_dist_olos_los)
    vehs.add_key('nlos', idxs_other_vehs[is_nlos])
    is_olos_los=np.invert(is_nlos)
    vehs.add_key('olos_los', idxs_other_vehs[is_olos_los])
    # utils.debug(debug, # time_start)

    # Determine OLOS and LOS
    # time_start = # utils.debug(debug, None, 'Determining OLOS and LOS')
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos=prop.veh_cons_are_olos(vehs.get_points('center'),
                                        vehs.get_points('olos_los'), margin=2)
    is_los=np.invert(is_olos)
    vehs.add_key('olos', vehs.get_idxs('olos_los')[is_olos])
    vehs.add_key('los', vehs.get_idxs('olos_los')[is_los])
    # utils.debug(debug, # time_start)
    d=min([vehs.get_points('center').distance(geo.Point(x[1]['x'],
                                                        x[1]['y']))
            for x in graph_streets_wave.node.items()])
    return (len(vehs.get_idxs("olos_los"))), (d)

       


def execute_main():
    sim_mode='single'  # 'single', 'multi', 'multiprocess'
    place='Upper West Side - New York - USA'
    place='Linz Stadt - Austria'
    # place='Neubau - Vienna - Austria'
    use_pathloss=False  # TODO: Implement functions to use use_pathloss
    which_result=1
    densities_veh=50e-3
    density_type='area'
    max_dist_olos_los=350
    max_dist_nlos=140
    iterations=1
    max_pl=150
    show_plot=False
    send_mail=False
    mail_to='thomas.blazek@nt.tuwien.ac.at'

    # TODO: temp!
    # place = 'Neubau - Wien - Austria'
    densities_veh=250e-6
    iterations=1
    # place = 'Neubau - Vienna - Austria'
    sim_mode='single'

    # Adapt static input parameters
    static_params={'place': place,
                     'which_result': which_result,
                     'density_type': density_type,
                     'max_dist_olos_los': max_dist_olos_los,
                     'max_dist_nlos': max_dist_nlos}

    # Switch to selected simulation mode
    print(sim_mode)
    if sim_mode == 'multi':
        net_connectivities=np.zeros([iterations, np.size(densities_veh)])
        for iteration in np.arange(iterations):
            for idx_density, density in enumerate(densities_veh):
                print('Densitiy: {:.2E}, Iteration: {:d}'.format(
                    density, iteration))
                net=load_network(static_params['place'],
                                   which_result=static_params['which_result'],
                                   debug=False)
                generate_vehicles(net, density_veh=density,
                                  density_type=static_params['density_type'],
                                  debug=False)
                net_connectivity, path_redundancy=main_sim_multi(net, max_dist_olos_los=max_dist_olos_los,
                                   max_dist_nlos=max_dist_nlos, debug=True)
                net_connectivities[iteration, idx_density]=net_connectivity
            # TODO: Adapt filename and saved variable structure from
            # multiprocess!
            np.save('results/net_connectivities',
                    net_connectivities[:iteration + 1])

        if show_plot:
            plot.plot_cluster_max(net['graph_streets'], net['gdf_buildings'],
                                  net['vehs'], show=False, place=place)
            plt.show()
    elif sim_mode == 'single':
        if np.size(densities_veh) > 1:
            raise ValueError(
                'Single simulation mode can only simulate 1 density value')

        net=load_network(static_params['place'],
                           which_result=static_params['which_result'],
                           debug=False)
        generate_vehicles(net, density_veh=densities_veh,
                          density_type=static_params['density_type'],
                          debug=False)
        with mp.Pool(12) as p:
            return_val = p.starmap(main_sim, zip(repeat(net),repeat(max_pl),
                                               repeat(True), repeat(300), range(net['vehs']  .count)))
        with open("return_vals.p","wb") as fp:
            pickle.dump(return_val, fp)
        

    else:
        raise NotImplementedError('Simulation type not supported')
if __name__ == '__main__':
    execute_main()
