""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

import os.path
import numpy as np
import pickle
# TODO: update osmnx and delete _git
import osmnx_git as ox
import matplotlib.pyplot as plt
import ipdb
import networkx as nx
import argparse

import shapely.geometry as geom

# TODO: edge['geometry'].length and edge['length'] are not equal!

def plot_streets_and_buildings(streets, buildings=None, show=True, filename=None, dpi=300):
    """ Plots streets and buildings"""

    # TODO: street width!
    # TODO: bug when plotting buildings, inner area not empty!
    fig, axi = ox.plot_graph(
        streets, show=False, close=False, node_size=0, dpi=dpi, edge_color='#333333')

    if buildings is not None:
        ox.plot_buildings(buildings, fig=fig, ax=axi,
                          show=False, close=False, dpi=dpi, color='#999999')

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    return fig, axi


def download_place(place, network_type='drive', file_prefix=None, which_result=1, project=True):
    """ Downloads streets and buildings for a place, saves the data to disk and returns them """

    if file_prefix is None:
        file_prefix = 'data/{}'.format(string_to_filename(place))

    # Streets
    streets = ox.graph_from_place(
        place, network_type=network_type, which_result=which_result)
    if project:
        streets = ox.project_graph(streets)
    filename_streets = '{}_streets.pickle'.format(file_prefix)
    pickle.dump(streets, open(filename_streets, 'wb'))

    # Buildings
    gdf = ox.gdf_from_place(place, which_result=which_result)
    polygon = gdf['geometry'].iloc[0]
    buildings = ox.create_buildings_gdf(polygon)
    if project:
        buildings = ox.project_gdf(buildings)
    filename_buildings = '{}_buildings.pickle'.format(file_prefix)
    pickle.dump(buildings, open(filename_buildings, 'wb'))

    # Return data
    data = {'streets': streets, 'buildings': buildings}
    return data


def load_place(file_prefix):
    """ Loads previously downloaded street and building data of a place"""
    filename_buildings = '{}_buildings.pickle'.format(file_prefix)
    buildings = pickle.load(open(filename_buildings, 'rb'))
    filename_streets = '{}_streets.pickle'.format(file_prefix)
    streets = pickle.load(open(filename_streets, 'rb'))
    place = {'streets': streets, 'buildings': buildings}
    return place


def string_to_filename(string):
    """ Cleans a string up to be used as a filename"""
    keepcharacters = ('_', '-')
    filename = ''.join(c for c in string if c.isalnum()
                       or c in keepcharacters).rstrip()
    return filename


def setup_debug():
    """ Sets osmnx up for a debugging session"""
    ox.config(log_console=True, use_cache=True)


def get_street_coordinates(streets):
    """ Returns the coordinates of the streets in a graph"""
    # TODO: use streets.number_of_edges() ?
    lines = []
    for u, v, data in streets.edges(data=True):
        if 'geometry' in data:
            coord_xs, coord_ys = data['geometry'].xy
            lines.append(list(zip(coord_xs, coord_ys)))
        else:
            coord_x1 = streets.node[u]['x']
            coord_y1 = streets.node[u]['y']
            coord_x2 = streets.node[v]['x']
            coord_y2 = streets.node[v]['y']
            line = [(coord_x1, coord_y1), (coord_x2, coord_y2)]
            lines.append(line)

    return lines


def add_geometry(streets):
    """ Adds geometry object to the edges of the graph where they are missing"""
    for u, v, data in streets.edges(data=True):
        if 'geometry' not in data:
            coord_x1 = streets.node[u]['x']
            coord_y1 = streets.node[u]['y']
            coord_x2 = streets.node[v]['x']
            coord_y2 = streets.node[v]['y']
            data['geometry'] = geom.LineString(
                [(coord_x1, coord_y1), (coord_x2, coord_y2)])


def check_geometry(streets):
    """ Checks if all edges of the graph have a geometry object"""
    complete = True
    for u, v, data in streets.edges(data=True):
        if 'geometry' not in data:
            complete = False
            break

    return complete


def line_intersects_streets(line, streets):
    """ Checks a if a line intersects with any of the streets"""
    # TODO: more efficiently? adjacency?
    intersects = False
    for u, v, data in streets.edges(data=True):
        if line.intersects(data['geometry']):
            intersects = True
            break

    return intersects


def line_intersects_buildings(line, buildings):
    """ Checks if a line intersects with any of the buildings"""
    intersects = False
    for geometry in buildings['geometry']:
        if line.intersects(geometry):
            intersects = True
            break

    return intersects


def line_intersects_points(line, points, margin=1):
    """ Checks if a line intersects with any of the points within a margin """

    intersects = False

    for point in points:
        proj = line.project(point)
        point_in_roi = (proj > 0) and (proj < line.length)
        distance_small = line.distance(point) < margin
        if point_in_roi and distance_small:
            intersects = True
            break

    return intersects


def veh_cons_are_nlos(point_own, point_vehs, buildings):
    """ Determines for each connection if it is NLOS or not"""

    is_nlos = np.zeros(np.size(point_vehs), dtype=bool)

    for index, point in np.ndenumerate(point_vehs):
        line = geom.LineString([point_own, point])
        is_nlos[index] = line_intersects_buildings(line, buildings)

    return is_nlos

def veh_cons_are_olos(point_own, point_vehs, margin=1):
    """ Determines for each LOS/OLOS connection if it is OLOS """

    #TODO: Also use NLOS vehicles!

    is_olos = np.zeros(np.size(point_vehs), dtype=bool)

    for index, point in np.ndenumerate(point_vehs):
        line = geom.LineString([point_own, point])
        indices_other = np.ones(np.size(point_vehs), dtype=bool)
        indices_other[index] = False
        is_olos[index] = line_intersects_points(line, point_vehs[indices_other], margin=margin)

    return is_olos


def get_street_lengths(streets):
    """ Returns the lengths of the streets in a graph"""
    # TODO: use streets.number_of_edges() ?
    lengths = []
    for u, v, data in streets.edges(data=True):
        lengths.append(data['length'])
    return lengths


def choose_random_streets(lengths, count=1):
    """ Chooses random streets with probabilities relative to their length"""
    total_length = sum(lengths)
    probs = lengths / total_length
    count_streets = np.size(lengths)
    indices = np.zeros(count, dtype=int)
    indices = np.random.choice(count_streets, size=count, p=probs)
    return indices


def choose_random_point(street, count=1):
    """Chooses random points along street """
    distances = np.random.random(count)
    points = np.zeros_like(distances, dtype=geom.Point)
    for index, dist in np.ndenumerate(distances):
        points[index] = street.interpolate(dist, normalized=True)

    return points


def extract_point_array(points):
    """Extracts coordinates form a point array """
    coords_x = np.zeros(np.size(points), dtype=float)
    coords_y = np.zeros(np.size(points), dtype=float)

    for index, point in np.ndenumerate(points):
        coords_x[index] = point.x
        coords_y[index] = point.y

    return coords_x, coords_y


def find_center_veh(coords_x, coords_y):
    """Finds the index of the vehicle at the center of the map """
    min_x = np.amin(coords_x)
    max_x = np.amax(coords_x)
    min_y = np.amin(coords_y)
    max_y = np.amax(coords_y)
    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2
    coords_center = np.array((mean_x, mean_y))
    coords_veh = np.vstack((coords_x, coords_y)).T
    distances_center = np.linalg.norm(
        coords_center - coords_veh, ord=2, axis=1)
    index_center_veh = np.argmin(distances_center)
    return index_center_veh


def main_test(place, which_result=1, count_veh=100):
    """ Test the whole functionality"""

    # Setup
    setup_debug()
    print('Running main test')

    # Load data
    print('Loading data')
    file_prefix = 'data/{}'.format(string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(
        string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(
        string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings):
        # Load from file
        print('LOADING FROM DISK')
        data = load_place(file_prefix)
    else:
        # Load from internet
        print('DOWNLOADING')
        data = download_place(place, which_result=which_result)

    # Plot streets and buildings
    plot_streets_and_buildings(data['streets'], data['buildings'], show=False, dpi=300)

    # Choose random streets and position on streets
    print('Choosing random vehicle positions')
    streets = data['streets']
    add_geometry(streets)
    street_lengths = get_street_lengths(streets)
    rand_index = choose_random_streets(street_lengths, count_veh)
    points = np.zeros(0, dtype=geom.Point)
    for index in rand_index:
        street_geom = streets.edges(data=True)[index][2]['geometry']
        point = choose_random_point(street_geom)
        points = np.append(points, point)
    x_coords, y_coords = extract_point_array(points)

    # Find center vehicle and plot
    print('Finding center vehicle')
    index_center_veh = find_center_veh(x_coords, y_coords)
    index_other_vehs = np.ones(len(points), dtype=bool)
    index_other_vehs[index_center_veh] = False
    x_coord_center_veh = x_coords[index_center_veh]
    y_coord_center_veh = y_coords[index_center_veh]
    x_coord_other_vehs = x_coords[index_other_vehs]
    y_coord_other_vehs = y_coords[index_other_vehs]
    point_center_veh = points[index_center_veh]
    points_other_veh = points[index_other_vehs]
    plt.scatter(x_coord_center_veh, y_coord_center_veh, label='Own', zorder=10)

    # Determine NLOS and OLOS/LOS
    print('Determining propagation condition')
    buildings = data['buildings']
    is_nlos = veh_cons_are_nlos(point_center_veh, points_other_veh, buildings)
    x_coord_nlos_vehs = x_coord_other_vehs[is_nlos]
    y_coord_nlos_vehs = y_coord_other_vehs[is_nlos]
    plt.scatter(x_coord_nlos_vehs, y_coord_nlos_vehs, label='NLOS', zorder=5, alpha=0.5)

    # Determine OLOS and LOS
    is_olos_los = np.invert(is_nlos)
    x_coord_olos_los_vehs = x_coord_other_vehs[is_olos_los]
    y_coord_olos_los_vehs = y_coord_other_vehs[is_olos_los]
    points_olos_los = points_other_veh[is_olos_los]
    # TODO: choose margin wisely
    is_olos = veh_cons_are_olos(point_center_veh, points_olos_los, margin=1.5)
    is_los = np.invert(is_olos)
    x_coord_olos_vehs = x_coord_olos_los_vehs[is_olos]
    y_coord_olos_vehs = y_coord_olos_los_vehs[is_olos]
    x_coord_los_vehs = x_coord_olos_los_vehs[is_los]
    y_coord_los_vehs = y_coord_olos_los_vehs[is_los]
    plt.scatter(x_coord_olos_vehs, y_coord_olos_vehs, label='OLOS', zorder=8, alpha=0.75)
    plt.scatter(x_coord_los_vehs, y_coord_los_vehs, label='LOS', zorder=8, alpha=0.75)

    # Show the plots
    print('Showing plot')
    plt.legend()
    plt.show()

def parse_arguments():
    """Parses the command line arguments and returns them """
    parser = argparse.ArgumentParser(description='Simulate connections on map.')
    parser.add_argument('-p', type=str, default='neubau - vienna - austria', help='place')
    parser.add_argument('-c', type=int, default=1000, help='number of vehicles')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = parse_arguments()
    main_test(args.p, which_result=1, count_veh=args.c)
