""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

import pickle
# TODO: update osmnx and delte _git
import osmnx_git as ox
import matplotlib.pyplot as plt
import networkx as nx


def plot_streets_and_buildings(streets, buildings=None, show=True, filename=None):
    """ Plots streets and buildings"""

    # TODO: street width!
    # TODO: bug when plotting buildings, inner area not empty!
    fig, axi = ox.plot_graph(streets, show=False, close=False, node_size=0)

    if buildings is not None:
        ox.plot_buildings(buildings, fig=fig, ax=axi, show=False, close=False)

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


# def get_street_coordinates(streets):
#     """ Returns the coordinates of a street graph"""
#     # TODO: rework, just 1st test version
#     for edge_keys in nx.edges_iter(streets):
#         for multi_key in streets[edge_keys[0]][edge_keys[1]]:
#             edge = streets[edge_keys[0]][edge_keys[1]][multi_key]
#             if 'geometry' in edge:
#                 coords = edge['geometry'].xy
#                 plt.plot(coords[0], coords[1], c='grey')
#             else:
#                 x1 = streets.node[edge_keys[0]]['x']
#                 y1 = streets.node[edge_keys[0]]['y']
#                 x2 = streets.node[edge_keys[1]]['x']
#                 y2 = streets.node[edge_keys[1]]['y']
#                 plt.plot((x1, x2), (y1, y2), c='grey')


def main_test(place, which_result=1):
    """ Test the functionality"""
    # TODO: temp, delete
    setup_debug()
    filename = string_to_filename('images/{}.pdf'.format(place))
    data = download_place(place, which_result=which_result)
    plot_streets_and_buildings(data['streets'], data['buildings'], show=False, filename=filename)

if __name__ == '__main__':
    main_test('vienna - austria', 2)
