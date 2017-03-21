""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

import os.path
import numpy as np
import pickle
# TODO: update osmnx and delete _git
import osmnx_git as ox
import matplotlib.pyplot as plt
# import networkx as nx

from shapely.geometry import LineString

# TODO: edge['geometry'].length and edge['length'] are not equal!

def plot_streets_and_buildings(streets, buildings=None, show=True, filename=None, dpi=300):
    """ Plots streets and buildings"""

    # TODO: street width!
    # TODO: bug when plotting buildings, inner area not empty!
    fig, axi = ox.plot_graph(streets, show=False, close=False, node_size=0, dpi=dpi)

    if buildings is not None:
        ox.plot_buildings(buildings, fig=fig, ax=axi, show=False, close=False, dpi=dpi)

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
            data['geometry'] = LineString([(coord_x1, coord_x2), (coord_y1, coord_y2)])

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
    # TODO: !
    return False


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
    probs = lengths/total_length
    count_streets = np.size(lengths)
    indices = np.zeros(count, dtype=int)
    indices = np.random.choice(count_streets, size=count, p=probs)
    return indices

def main_test(place, which_result=1):
    """ Test the functionality"""
    # TODO: temp, delete
    setup_debug()
    print('RUNNING MAIN TEST')
    
    # Load data
    file_prefix = 'data/{}'.format(string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings):
        print('LOADING FROM DISK')
        data = load_place(file_prefix)
    else:
        print('DOWNLOADING')
        data = download_place(place, which_result=which_result)

    # Plot
    filename_img = 'images/{}.pdf'.format(string_to_filename(place))
    plot_streets_and_buildings(data['streets'], data['buildings'], show=False, dpi=300)
        # filename=filename_img)

    # Test intersection and random functions
    streets = data['streets']
    street_lengths = get_street_lengths(streets)
    rand_index = choose_random_streets(street_lengths, 2)
    line_street_1 = streets.edges(data=True)[rand_index[0]][2]['geometry']
    line_street_2 = streets.edges(data=True)[rand_index[1]][2]['geometry']
    plt.scatter(line_street_1.xy[0], line_street_1.xy[1])
    plt.scatter(line_street_2.xy[0], line_street_2.xy[1])
    
    add_geometry(streets)
    intersects_1 = line_intersects_streets(line_street_1, streets)
    if intersects_1:
        print('INTERSECT')
    else:
        print('NO INTERSECT')
    
    plt.show()

if __name__ == '__main__':
    main_test('innere stadt - vienna - austria', 1)
