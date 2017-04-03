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

class Vehicles:
    # TODO: only points as attributes and get coordinates from points when requested?

    def __init__(self, points, graphs=None):
        self.points = points
        self.coordinates = geom_o.extract_point_array(points)
        self.graphs = graphs
        self.pathlosses = np.zeros(np.size(points))
        self.distances = np.zeros(np.size(points))
        self.idxs = {}

    def add_key(self, key, value):
        self.idxs[key] = value

    def get(self, key=None):
        if key is None:
            return self.coordinates
        else:
            return self.coordinates[self.idxs[key]]

    def get_points(self, key=None):
        if key is None:
            return self.points
        else:
            return self.points[self.idxs[key]]

    def get_graph(self, key=None):
        if key is None:
            return self.graphs
        else:
            return self.graphs[self.idxs[key]]

    def get_idxs(self, key):
        return self.idxs[key]

    def set_pathlosses(self, key, values):
        self.pathlosses[self.idxs[key]] = values

    def get_pathlosses(self, key=None):
        if key is None:
            return self.pathlosses
        else:
            return self.pathlosses[self.idxs[key]]

    def set_distances(self, key, values):
        self.distances[self.idxs[key]] = values

    def get_distances(self, key=None):
        if key is None:
            return self.distances
        else:
            return self.distances[self.idxs[key]]

def generate_vehs(graph_streets, street_idxs):

    count_veh = np.size(street_idxs)
    points_vehs = np.zeros(count_veh, dtype=object)
    graphs_vehs = np.zeros(count_veh, dtype=object)

    for iteration, index in enumerate(street_idxs):
        street = graph_streets.edges(data=True)[index]
        street_geom = street[2]['geometry']
        point_veh = dist.choose_random_point(street_geom)
        points_vehs[iteration] = point_veh[0]
        # NOTE: All vehicle nodes get the prefix 'v'
        node = 'v' + str(iteration)
        # Add vehicle, needed intersections and edges to graph
        graph_iter = nx.MultiGraph(node_veh=node)
        node_attr = {'geometry': point_veh[0], 'x' : point_veh[0].x, 'y' : point_veh[0].y}
        graph_iter.add_node(node, attr_dict=node_attr)
        graph_iter.add_nodes_from(street[0:2])

        # Determine street parts that connect vehicle to intersections
        street_before, street_after = geom_o.split_line_at_point(street_geom, point_veh[0])
        edge_attr = {'geometry': street_before, 'length': street_before.length, 'is_veh_edge': True}
        graph_iter.add_edge(node, street[0], attr_dict=edge_attr)
        edge_attr = {'geometry': street_after, 'length': street_after.length, 'is_veh_edge': True}
        graph_iter.add_edge(node, street[1], attr_dict=edge_attr)

        # Copy the created graph
        graphs_vehs[iteration] = graph_iter.copy()

    vehs = Vehicles(points_vehs, graphs_vehs)
    return vehs

def main_sim(place, which_result=1, density_veh=100, use_pathloss=True, max_pl=150, density_type='absolute',
             debug=False):
    """ Test the whole functionality"""

    # Setup
    ox_a.setup(debug)

    # Load data
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(utils.string_to_filename(place))
    filename_data_boundary = 'data/{}_boundary.pickle'.format(utils.string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings) and \
            os.path.isfile(filename_data_boundary):
        # Load from file
        time_start = utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        time_start = utils.debug(debug, None, 'Loading data from the internet')
        data = ox_a.download_place(place, which_result=which_result)

    utils.debug(debug, time_start)

    # Choose random streets and position on streets
    time_start = utils.debug(debug, None, 'Building graph for wave propagation')

    graph_streets = data['streets']
    gdf_buildings = data['buildings']

    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    ox_a.add_geometry(graph_streets)
    streets_wave = graph_streets.to_undirected()
    prop.add_edges_if_los(streets_wave, gdf_buildings)

    utils.debug(debug, time_start)

    # Streets and positions selection
    time_start = utils.debug(debug, None, 'Choosing random vehicle positions')

    street_lengths = geom_o.get_street_lengths(graph_streets)

    if density_type == 'absolute':
        count_veh = int(density_veh)
    elif density_type == 'length':
        count_veh = int(round(density_veh*np.sum(street_lengths)))
    elif density_type == 'area':
        area = data['boundary'].area
        count_veh = int(round(density_veh*area))
    else:
        raise ValueError('Density type not supported')

    rand_street_idxs = dist.choose_random_streets(street_lengths, count_veh)
    utils.debug(debug, time_start)

    # Vehicle generation
    time_start = utils.debug(debug, None, 'Generating vehicles')
    vehs = generate_vehs(graph_streets, rand_street_idxs)
    utils.debug(debug, time_start)

    # Find center vehicle
    time_start = utils.debug(debug, None, 'Finding center vehicle')

    index_center_veh = geom_o.find_center_veh(vehs.get())
    index_other_vehs = np.where(np.arange(count_veh) != index_center_veh)[0]
    vehs.add_key('center', index_center_veh)
    vehs.add_key('other', index_other_vehs)

    utils.debug(debug, time_start)

    # Determine NLOS and OLOS/LOS
    time_start = utils.debug(debug, None, 'Determining propagation conditions')

    is_nlos = prop.veh_cons_are_nlos(vehs.get_points('center'),
                                     vehs.get_points('other'), gdf_buildings)
    vehs.add_key('nlos', index_other_vehs[is_nlos])  
    is_olos_los = np.invert(is_nlos)
    vehs.add_key('olos_los', index_other_vehs[is_olos_los])

    utils.debug(debug, time_start)


    if use_pathloss:
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
        time_start = utils.debug(debug, None, 'Determining orthogonal and parallel')

        is_orthogonal, coords_intersections = \
            prop.check_if_cons_orthogonal(streets_wave,
                                        vehs.get_graph('center'),
                                        vehs.get_graph('nlos'),
                                        max_angle=np.pi)
        is_paralell = np.invert(is_orthogonal)
        vehs.add_key('orth', vehs.get_idxs('nlos')[is_orthogonal])
        vehs.add_key('par', vehs.get_idxs('nlos')[is_paralell])

        utils.debug(debug, time_start)

        # Determining pathlosses for LOS and OLOS
        time_start = utils.debug(debug, None, 'Calculating pathlosses for OLOS and LOS')

        p_loss = pathloss.Pathloss()
        distances_olos_los = np.sqrt( \
            (vehs.get('olos_los')[:, 0] - vehs.get('center')[0])**2 + \
            (vehs.get('olos_los')[:, 1] - vehs.get('center')[1])**2)

        # TODO: why - ? fix in pathloss.py
        pathlosses_olos = -p_loss.pathloss_olos(distances_olos_los[is_olos])
        vehs.set_pathlosses('olos', pathlosses_olos)
        # TODO: why - ? fix in pathloss.py
        pathlosses_los = -p_loss.pathloss_los(distances_olos_los[is_los])
        vehs.set_pathlosses('los', pathlosses_los)

        utils.debug(debug, time_start)

        # Determining pathlosses for NLOS orthogonal
        time_start = utils.debug(debug, None, 'Calculating pathlosses for NLOS orthogonal')

        # NOTE: Assumes center vehicle is receiver
        # NOTE: Uses airline vehicle -> intersection -> vehicle and not street route
        distances_orth_tx = np.sqrt(
            (vehs.get('orth')[:, 0] - coords_intersections[is_orthogonal, 0])**2 +
            (vehs.get('orth')[:, 1] - coords_intersections[is_orthogonal, 1])**2)

        distances_orth_rx = np.sqrt(
            (vehs.get('center')[0] - coords_intersections[is_orthogonal, 0])**2 +
            (vehs.get('center')[1] - coords_intersections[is_orthogonal, 1])**2)

        pathlosses_orth = p_loss.pathloss_nlos(distances_orth_rx, distances_orth_tx)
        vehs.set_pathlosses('orth', pathlosses_orth)
        pathlosses_par = np.Infinity*np.ones(np.sum(is_paralell))
        vehs.set_pathlosses('par', pathlosses_par)
        
        utils.debug(debug, time_start)

        # Determine in range / out of range
        time_start = utils.debug(debug, None, 'Determining in range vehicles')

        index_in_range = vehs.get_pathlosses('other') < max_pl
        index_out_range = np.invert(index_in_range)
        vehs.add_key('in_range', vehs.get_idxs('other')[index_in_range])
        vehs.add_key('out_range', vehs.get_idxs('other')[index_out_range])

        utils.debug(debug, time_start)

        # Plots specific to pathloss method
        plot.plot_prop_cond(graph_streets, gdf_buildings, vehs, show=False, place=place)
        plot.plot_pathloss(graph_streets, gdf_buildings, vehs, show=False, place=place)


    else: # not use_pathloss
        # TODO: make parameters
        max_dist_los = 250
        max_dist_nlos = 140
        distances = np.linalg.norm(vehs.get('other') - vehs.get('center'), ord=2, axis=1)
        vehs.set_distances('other', distances)
        index_in_range_los = vehs.get_distances('olos_los') < max_dist_los
        index_in_range_nlos = vehs.get_distances('nlos') < max_dist_nlos

        ipdb.set_trace()

    # Common plots for both methods
    plot.plot_con_status(graph_streets, gdf_buildings, vehs, show=False, place=place)
    plt.show()




def parse_arguments():
    """Parses the command line arguments and returns them """
    parser = argparse.ArgumentParser(description='Simulate vehicle connections on map')
    parser.add_argument('-p', type=str, default='Neubau - Vienna - Austria', help='place')
    parser.add_argument('-w', type=int, default=1, help='which result')
    parser.add_argument('-d', type=float, default=1000, help='vehicle density')
    parser.add_argument('-s', type=int, default=1, help='flag determining pathloss usage (instead of distance)')
    parser.add_argument('-l', type=float, default=150, help='pathloss threshold [dB]')
    parser.add_argument('-t', type=str, default='absolute',
                        help='density type (absolute, length, area)')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = parse_arguments()
    main_sim(args.p, which_result=args.w, density_veh=args.d, use_pathloss=bool(args.s), max_pl=args.l, density_type=args.t,
             debug=True)
