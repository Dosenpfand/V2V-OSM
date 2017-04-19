""" Generates streets, buildings and vehicles from OpenStreetMap data
with osmnx"""

# Standard imports
import os.path
import os
# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import networkx as nx
import pickle
# Local imports
# import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles
import shapely.geometry as geo
import propagation as prop
# https://www.cs.jhu.edu/~xinjin/files/IVCS11_VANET.pdf


def prepare_network(place,
                    which_result: float=1,
                    density_veh: float=100,
                    density_type: str='absolute',
                    debug: bool=False) -> dict:
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

    if (os.path.isfile(filename_data_streets) and
            os.path.isfile(filename_data_buildings) and
            os.path.isfile(filename_data_boundary)):
        # Load from file

        time_start = utils.debug(None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        time_start = utils.debug(None, 'Loading data from the internet')
        data = ox_a.download_place(place, which_result=which_result)

    utils.debug(time_start)

    # Choose random streets and position on streets
    time_start = utils.debug(None, 'Building graph for wave propagation')

    graph_streets = data['streets']
    gdf_buildings = data['buildings']

    # Vehicles are placed in a undirected version of the graph
    # because electromagnetic
    # waves do not respect driving directions
    ox_a.add_geometry(graph_streets)
    graph_streets_wave = graph_streets.to_undirected()
    prop.add_edges_if_los(graph_streets_wave, gdf_buildings)

    utils.debug(time_start)

    # Streets and positions selection
    time_start = utils.debug(None, 'Choosing random vehicle positions')

    street_lengths = geom_o.get_street_lengths(graph_streets)

    if density_type == 'absolute':
        count_veh = int(density_veh)
    elif density_type == 'length':
        count_veh = int(round(density_veh * np.sum(street_lengths)))
    elif density_type == 'area':
        area = data['boundary'].area
        count_veh = int(round(density_veh * area))
    elif density_type == 'external':
        pass
    else:
        raise ValueError('Density type not supported')
    if density_type != 'external':
        rand_street_idxs = vehicles.choose_random_streets(
            street_lengths, count_veh)
        vehicular_point_coords = None
    else:
        with open(os.path.join(os.getcwd(), 'lnz_unrouted_3000v_2500ms.p'),
                  'rb') as fp:
            results_dict = pickle.load(fp)
        vals = results_dict["networks"][300]
        vehicular_point_coords = [geo.Point(coords[0], coords[1])
                                  for coords in zip(vals['x'], vals['y'])]
        rand_street_idxs = None
    utils.debug(time_start)

    # Vehicle generation
    time_start = utils.debug(None, 'Generating vehicles')

    vehs = vehicles.generate_vehs(
        graph_streets, rand_street_idxs, vehicular_point_coords)

    utils.debug(time_start)

    network = {'graph_streets': graph_streets,
               'graph_streets_wave': graph_streets_wave,
               'gdf_buildings': gdf_buildings, 'vehs': vehs}

    return network


def main_sim_multi(network, max_dist_olos_los=250,
                   max_dist_nlos=140, debug=False):
    """Simulates all combinations of connections using
     only distances (not pathlosses) """

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
    if debug:
        print('Network connectivity: {:.2f} %'.format(
            net_connectivity * 100))

    return net_connectivity


def main_sim(network, max_pl=150, debug=False):
    """Simulates the connections from one to all other
    vehicles using pathloss functions """

    # Initialize
    vehs = network['vehs']
    # graph_streets_wave = network['graph_streets_wave']
    # gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    vehs.allocate(count_veh)

    # Find center vehicle
    # time_start = utils.debug(None, 'Finding center vehicle')
    idx_center_veh = geom_o.find_center_veh(vehs.get())
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)


def setup():
    place = 'Neubau - Wien - Austria'
    place = 'Linz Stadt - Austria'
    which_result = 1
    density_veh = 5e-6
    density_type = 'external'
    # max_dist_olos_los = 250
    # max_dist_nlos = 140
    # iterations = 10
    max_pl = 150

    # MULTI
    # net_connectivities = np.zeros(iterations)
    # SINGLE
    net = prepare_network(place,
                          which_result=which_result,
                          density_veh=density_veh,
                          density_type=density_type, debug=True)
    main_sim(net, max_pl=max_pl, debug=True)
    # Plots
    print()
    print(net["vehs"])
    print(net["vehs"])
    plot.plot_streets_and_buildings(net['graph_streets'],
                                    net['gdf_buildings'],
                                    show=False)
    print("scatter:")
    vehs = net["vehs"].get("all")

    plt.scatter(vehs[:, 0], vehs[:, 1], label='Own',
                marker='x', zorder=10,
                s=2 * plt.rcParams['lines.markersize']**2, c='black')
    plt.show()


if __name__ == '__main__':
    setup()
