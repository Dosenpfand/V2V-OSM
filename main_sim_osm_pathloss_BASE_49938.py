""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import argparse
import os.path
import ipdb

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicle_distribution as dist
import propagation as prop

def main_sim(place, which_result=1, count_veh=100, max_pl=100, debug=False):
    """ Test the whole functionality"""

    # Setup
    ox_a.setup(debug)

    # Load data
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(
        utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(
        utils.string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings):
        # Load from file
        time_start = utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        time_start = utils.debug(debug, True, 'Loading data from the internet')
        data = ox_a.download_place(place, which_result=which_result)

    utils.debug(debug, time_start)

    # Choose random streets and position on streets
    time_start = utils.debug(debug, None, 'Building graph for wave propagation')

    streets = data['streets']
    buildings = data['buildings']

    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    ox_a.add_geometry(streets)
    streets_wave = streets.to_undirected()
    prop.add_edges_if_los(streets_wave, buildings)

    utils.debug(debug, time_start)
    time_start = utils.debug(debug, None, 'Choosing random vehicle positions')

    street_lengths = geom_o.get_street_lengths(streets)
    rand_index = dist.choose_random_streets(street_lengths, count_veh)
    points = np.zeros(count_veh, dtype=object)

    utils.debug(debug, time_start)
    time_start = utils.debug(debug, None, 'Creating graphs for vehicles')

    graphs_veh = np.zeros(count_veh, dtype=object)
    for iteration, index in enumerate(rand_index):
        street = streets.edges(data=True)[index]
        street_geom = street[2]['geometry']
        point = dist.choose_random_point(street_geom)
        points[iteration] = point[0]
        # NOTE: All vehicle nodes get the prefix 'v'
        node = 'v' + str(iteration)
        # Add vehicle, needed intersections and edges to graph
        graph_iter = nx.MultiGraph(node_veh=node)
        node_attr = {'geometry': point[0], 'x' : point[0].x, 'y' : point[0].y}
        graph_iter.add_node(node, attr_dict=node_attr)
        graph_iter.add_nodes_from(street[0:1])

        street_before, street_after = geom_o.split_line_at_point(street_geom, point[0])
        street_length = street_before.length
        edge_attr = {'geometry': street_before, 'length': street_length, 'is_veh_edge': True}
        graph_iter.add_edge(node, street[0], attr_dict=edge_attr)
        street_length = street_after.length
        edge_attr = {'geometry': street_after, 'length': street_length, 'is_veh_edge': True}
        graph_iter.add_edge(node, street[1], attr_dict=edge_attr)

        graphs_veh[iteration] = graph_iter.copy()

    coords_vehs = {}
    coords_vehs['all'] = geom_o.extract_point_array(points)

    utils.debug(debug, time_start)

    # Find center vehicle
    time_start = utils.debug(debug, None, 'Finding center vehicle')

    index_center_veh = geom_o.find_center_veh(coords_vehs['all'])
    index_other_vehs = np.ones(len(points), dtype=bool)
    index_other_vehs[index_center_veh] = False
    coords_vehs['center'] = coords_vehs['all'][index_center_veh, :]
    coords_vehs['other'] = coords_vehs['all'][index_other_vehs, :]
    point_center_veh = points[index_center_veh]
    points_other_veh = points[index_other_vehs]

    utils.debug(debug, time_start)

    # Determine NLOS and OLOS/LOS
    time_start = utils.debug(debug, None, 'Determining propagation conditions')

    is_nlos = prop.veh_cons_are_nlos(point_center_veh, points_other_veh, buildings)
    coords_vehs['nlos'] = coords_vehs['other'][is_nlos, :]

    utils.debug(debug, time_start)

    # Determine OLOS and LOS
    time_start = utils.debug(debug, None, 'Determining OLOS and LOS')

    is_olos_los = np.invert(is_nlos)
    coords_vehs['olos_los'] = coords_vehs['other'][is_olos_los, :]
    points_olos_los = points_other_veh[is_olos_los]
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos = prop.veh_cons_are_olos(point_center_veh, points_olos_los, margin=2)
    is_los = np.invert(is_olos)
    coords_vehs['olos'] = coords_vehs['olos_los'][is_olos, :]
    coords_vehs['los'] = coords_vehs['olos_los'][is_los, :]

    utils.debug(debug, time_start)

    # Determine orthogonal and parallel
    time_start = utils.debug(debug, None, 'Determining orthogonal and parallel')

    graphs_veh_nlos = graphs_veh[index_other_vehs][is_nlos]
    graph_veh_own = graphs_veh[index_center_veh]
    is_orthogonal, coords_intersections = prop.check_if_cons_orthogonal(streets_wave,
                                                                        graph_veh_own,
                                                                        graphs_veh_nlos,
                                                                        max_angle=np.pi)
    is_paralell = np.invert(is_orthogonal)
    coords_vehs['orth'] = coords_vehs['nlos'][is_orthogonal, :]
    coords_vehs['par'] = coords_vehs['nlos'][is_paralell, :]

    utils.debug(debug, time_start)

    plot.plot_prop_cond(streets, buildings, coords_vehs, show=False, place=place)

    # Determining pathlosses for LOS and OLOS
    time_start = utils.debug(debug, None, 'Calculating pathlosses for OLOS and LOS')

    p_loss = pathloss.Pathloss()
    distances_olos_los = np.sqrt( \
        (coords_vehs['olos_los'][:, 0] - coords_vehs['center'][0])**2 + \
        (coords_vehs['olos_los'][:, 1] - coords_vehs['center'][1])**2)

    pathlosses_olos = p_loss.pathloss_olos(distances_olos_los[is_olos])
    pathlosses_los = p_loss.pathloss_los(distances_olos_los[is_los])

    pathlosses_olos_los = np.zeros(np.size(distances_olos_los))
    pathlosses_olos_los[is_olos] = pathlosses_olos
    pathlosses_olos_los[is_los] = pathlosses_los

    utils.debug(debug, time_start)

    # Determining pathlosses for NLOS orthogonal
    time_start = utils.debug(debug, None, 'Calculating pathlosses for NLOS orthogonal')

    # NOTE: Assumes center vehicle is receiver
    # NOTE: Uses airline vehicle -> intersection -> vehicle and not street route
    distances_orth_tx = np.sqrt(
        (coords_vehs['orth'][:, 0] - coords_intersections[is_orthogonal, 0])**2 +
        (coords_vehs['orth'][:, 1] - coords_intersections[is_orthogonal, 1])**2)

    distances_orth_rx = np.sqrt(
        (coords_vehs['center'][0] - coords_intersections[is_orthogonal, 0])**2 +
        (coords_vehs['center'][1] - coords_intersections[is_orthogonal, 1])**2)

    pathlosses_orth = p_loss.pathloss_nlos(distances_orth_rx, distances_orth_tx)

    pathlosses_nlos = np.zeros(np.shape(coords_vehs['nlos'])[0])
    pathlosses_nlos[is_paralell] = np.Infinity*np.ones(np.sum(is_paralell))
    pathlosses_nlos[is_orthogonal] = pathlosses_orth

    # Build complete pathloss array
    pathlosses = np.zeros(count_veh-1)
    # TODO: Why - ? Fix in pathloss.py
    pathlosses[is_olos_los] = -pathlosses_olos_los
    pathlosses[is_nlos] = pathlosses_nlos

    utils.debug(debug, time_start)

    plot.plot_pathloss(streets, buildings, coords_vehs, pathlosses, show=False, place=place)

    # Determine in range / out of range
    time_start = utils.debug(debug, None, 'Determining in range vehicles')

    index_in_range = pathlosses < max_pl
    index_out_range = np.invert(index_in_range)
    coords_vehs['in_range'] = coords_vehs['other'][index_in_range, :]
    coords_vehs['out_range'] = coords_vehs['other'][index_out_range, :]

    utils.debug(debug, time_start)
    plot.plot_con_status(streets, buildings, coords_vehs, show=False, place=place)

    # Show the plots
    if debug:
        print('Showing plot')
    plt.show()

def parse_arguments():
    """Parses the command line arguments and returns them """
    parser = argparse.ArgumentParser(description='Simulate vehicle connections on map')
    parser.add_argument('-p', type=str, default='Neubau - Vienna - Austria', help='place')
    parser.add_argument('-c', type=int, default=1000, help='number of vehicles')
    parser.add_argument('-w', type=int, default=1, help='which result')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = parse_arguments()
    main_sim(args.p, which_result=args.w, count_veh=args.c, max_pl=150, debug=True)