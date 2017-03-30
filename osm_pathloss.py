""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import time
import argparse
import os.path
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

import ipdb


def main_test(place, which_result=1, count_veh=100, max_pl=100, debug=False):
    """ Test the whole functionality"""

    # Setup
    ox_a.setup(debug)
    if debug:
        print('RUNNING MAIN SIMULATION')

    # Load data
    if debug:
        time_start = time.process_time()
        time_start_tot = time_start
        utils.print_nnl('Loading data')
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(
        utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(
        utils.string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings):
        # Load from file
        if debug:
            utils.print_nnl('from disk:')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        if debug:
            utils.print_nnl('from the internet:')
        data = ox_a.download_place(place, which_result=which_result)

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    # Choose random streets and position on streets
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Building graph for wave propagation:')
    streets = data['streets']
    buildings = data['buildings']

    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    ox_a.add_geometry(streets)
    streets_wave = streets.to_undirected()
    prop.add_edges_if_los(streets_wave, buildings)
    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    if debug:
        time_start = time.process_time()
        utils.print_nnl('Choosing random vehicle positions:')
    street_lengths = geom_o.get_street_lengths(streets)
    rand_index = dist.choose_random_streets(street_lengths, count_veh)
    points = np.zeros(count_veh, dtype=object)

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    if debug:
        time_start = time.process_time()
        utils.print_nnl('Creating graphs for vehicles:')

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

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    # Find center vehicle
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Finding center vehicle:')

    index_center_veh = geom_o.find_center_veh(coords_vehs['all'])
    index_other_vehs = np.ones(len(points), dtype=bool)
    index_other_vehs[index_center_veh] = False
    coords_vehs['center'] = coords_vehs['all'][index_center_veh, :]
    coords_vehs['other'] = coords_vehs['all'][index_other_vehs, :]
    point_center_veh = points[index_center_veh]
    points_other_veh = points[index_other_vehs]

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    # Determine NLOS and OLOS/LOS
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Determining propagation condition:')
    is_nlos = prop.veh_cons_are_nlos(point_center_veh, points_other_veh, buildings)
    coords_vehs['nlos'] = coords_vehs['other'][is_nlos, :]

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    # Determine OLOS and LOS
    if debug:
        utils.print_nnl('Determining OLOS and LOS:')
        time_start = time.process_time()
    is_olos_los = np.invert(is_nlos)
    coords_vehs['olos_los'] = coords_vehs['other'][is_olos_los, :]
    points_olos_los = points_other_veh[is_olos_los]
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos = prop.veh_cons_are_olos(point_center_veh, points_olos_los, margin=2)
    is_los = np.invert(is_olos)
    coords_vehs['olos'] = coords_vehs['olos_los'][is_olos, :]
    coords_vehs['los'] = coords_vehs['olos_los'][is_los, :]

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    # Determine orthogonal and parallel
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Determining orthogonal and parallel:')

    graphs_veh_nlos = graphs_veh[index_other_vehs][is_nlos]
    graph_veh_own = graphs_veh[index_center_veh]
    is_orthogonal, coords_intersections = prop.check_if_cons_orthogonal(streets_wave,
                                                                        graph_veh_own,
                                                                        graphs_veh_nlos,
                                                                        max_angle=np.pi)
    is_paralell = np.invert(is_orthogonal)
    coords_vehs['orth'] = coords_vehs['nlos'][is_orthogonal, :]
    coords_vehs['par'] = coords_vehs['nlos'][is_paralell, :]

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    plot.plot_prop_cond(streets, buildings, coords_vehs, show=False, place=place)

    # Determining pathlosses for LOS and OLOS
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Determining pathlosses for LOS and OLOS:')

    p_loss = pathloss.Pathloss()
    distances_olos_los = np.sqrt( \
        (coords_vehs['olos_los'][:, 0] - coords_vehs['center'][0])**2 + \
        (coords_vehs['olos_los'][:, 1] - coords_vehs['center'][1])**2)

    pathlosses_olos = p_loss.pathloss_olos(distances_olos_los[is_olos])
    pathlosses_los = p_loss.pathloss_los(distances_olos_los[is_los])

    pathlosses_olos_los = np.zeros(np.size(distances_olos_los))
    pathlosses_olos_los[is_olos] = pathlosses_olos
    pathlosses_olos_los[is_los] = pathlosses_los

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    # Determining pathlosses for NLOS orthogonal
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Determining pathlosses for NLOS orthogonal:')

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

    if debug:
        time_diff = time.process_time() - time_start
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))

    plot.plot_pathloss(streets, buildings, coords_vehs, pathlosses, show=False, place=place)

    # Determine in range / out of range
    # Determining pathlosses for NLOS orthogonal
    if debug:
        time_start = time.process_time()
        utils.print_nnl('Determining in range vehicles:')

    index_in_range = pathlosses < max_pl
    index_out_range = np.invert(index_in_range)
    coords_vehs['in_range'] = coords_vehs['other'][index_in_range, :]
    coords_vehs['out_range'] = coords_vehs['other'][index_out_range, :]

    if debug:
        time_diff = time.process_time() - time_start
        time_diff_tot = time.process_time() - time_start_tot
        utils.print_nnl(' {:.3f} seconds\n'.format(time_diff))
        utils.print_nnl('TOTAL RUNNING TIME: {:.3f} seconds\n'.format(time_diff_tot))

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
    main_test(args.p, which_result=args.w, count_veh=args.c, max_pl=150, debug=True)
