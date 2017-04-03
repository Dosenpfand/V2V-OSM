""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
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


def main_sim(place, which_result=1, count_veh=1000, max_pl=100, debug=False):
    """ Test the whole functionality"""

    # Setup
    ox_a.setup(debug)

    # Load data
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(
        utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(
        utils.string_to_filename(place))

    if (os.path.isfile(filename_data_streets) and
            os.path.isfile(filename_data_buildings)):
            # Load from file
        time_start = utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        time_start = utils.debug(debug, True, 'Loading data from the internet')
        data = ox_a.download_place(place, which_result=which_result)

    utils.debug(debug, time_start)

    # Choose random streets and position on streets
    time_start = utils.debug(
        debug, None, 'Building graph for wave propagation')

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
        node_attr = {'geometry': point[0], 'x': point[0].x, 'y': point[0].y}
        graph_iter.add_node(node, attr_dict=node_attr)
        graph_iter.add_nodes_from(street[0:1])

        street_before, street_after = geom_o.split_line_at_point(
            street_geom, point[0])
        street_length = street_before.length
        edge_attr = {'geometry': street_before,
                     'length': street_length, 'is_veh_edge': True}
        graph_iter.add_edge(node, street[0], attr_dict=edge_attr)
        street_length = street_after.length
        edge_attr = {'geometry': street_after,
                     'length': street_length, 'is_veh_edge': True}
        graph_iter.add_edge(node, street[1], attr_dict=edge_attr)

        graphs_veh[iteration] = graph_iter.copy()

    coords_vehs = {}
    coords_vehs['all'] = geom_o.extract_point_array(points)

    utils.debug(debug, time_start)

    # Find center vehicle
    time_start = utils.debug(debug, None, 'Finding center vehicle')

    index_center_veh = geom_o.find_center_veh(coords_vehs['all'])
    all_coords = coords_vehs['all']
    index_other_vehs = np.ones(len(points), dtype=bool)
    index_other_vehs[index_center_veh] = False
    coords_vehs['center'] = coords_vehs['all'][index_center_veh, :]
    coords_vehs['other'] = coords_vehs['all'][index_other_vehs, :]
    point_center_veh = points[index_center_veh]
    points_other_veh = points[index_other_vehs]

    utils.debug(debug, time_start)

    fig, axi = plot.plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)
    plt.scatter(coords_vehs["all"][:, 0], coords_vehs["all"][:, 1])
    plt.figure()
    sq_values = (np.array(coords_vehs['other']) -
                 np.array(coords_vehs['center']))**2
    ds = np.sqrt(np.sum(sq_values, 1))
    out_data, out_edges = np.histogram(ds, bins=100, density=True)
    out_data = out_data / np.sum(out_data)
    centroids = (out_edges[:-1] + out_edges[1:]) / 2
    plt.plot(centroids, np.cumsum(out_data))
    plt.show()


def parse_arguments():
    """Parses the command line arguments and returns them """
    parser = argparse.ArgumentParser(
        description='Simulate vehicle connections on map')
    parser.add_argument(
        '-p', type=str, default='Neubau - Vienna - Austria', help='place')
    parser.add_argument('-c', type=int, default=10000,
                        help='number of vehicles')
    parser.add_argument('-w', type=int, default=2, help='which result')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_arguments()
    main_sim(args.p, which_result=args.w,
             count_veh=args.c, max_pl=150, debug=True)
