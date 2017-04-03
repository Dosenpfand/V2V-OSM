""" Additional functions missing in the OSMnx package"""

import pickle
import networkx as nx
import shapely.ops as ops
import shapely.geometry as geom
import osmnx_git as ox # TODO: update osmnx and delete _git
import utils

def setup(debug=False):
    """ Sets osmnx up"""
    if debug:
        ox.config(log_console=True, use_cache=True)
    else:
        ox.config(log_console=False, use_cache=False)

def download_place(place, network_type='drive', file_prefix=None, which_result=1, project=True):
    """ Downloads streets and buildings for a place, saves the data to disk and returns them """

    if file_prefix is None:
        file_prefix = 'data/{}'.format(utils.string_to_filename(place))

    # Streets
    streets = ox.graph_from_place(
        place, network_type=network_type, which_result=which_result)
    if project:
        streets = ox.project_graph(streets)
    filename_streets = '{}_streets.pickle'.format(file_prefix)
    pickle.dump(streets, open(filename_streets, 'wb'))

    # Boundary and buildings
    boundary = ox.gdf_from_place(place, which_result=which_result)
    polygon = boundary['geometry'].iloc[0]
    buildings = ox.create_buildings_gdf(polygon)
    if project:
        buildings = ox.project_gdf(buildings)
        boundary = ox.project_gdf(boundary)

    # Save buildings
    filename_buildings = '{}_buildings.pickle'.format(file_prefix)
    pickle.dump(buildings, open(filename_buildings, 'wb'))

    # Save boundary
    filename_boundary = '{}_boundary.pickle'.format(file_prefix)
    pickle.dump(boundary, open(filename_boundary, 'wb'))

    # Return data
    data = {'streets': streets, 'buildings': buildings}
    return data

def load_place(file_prefix):
    """ Loads previously downloaded street and building data of a place"""

    filename_buildings = '{}_buildings.pickle'.format(file_prefix)
    buildings = pickle.load(open(filename_buildings, 'rb'))
    filename_streets = '{}_streets.pickle'.format(file_prefix)
    streets = pickle.load(open(filename_streets, 'rb'))
    filename_boundary = '{}_boundary.pickle'.format(file_prefix)
    boundary = pickle.load(open(filename_boundary, 'rb'))

    place = {'streets': streets, 'buildings': buildings, 'boundary': boundary}
    return place


def add_geometry(streets):
    """ Adds geometry object to the edges of the graph where they are missing"""
    for u_node, v_node, data in streets.edges(data=True):
        if 'geometry' not in data:
            coord_x1 = streets.node[u_node]['x']
            coord_y1 = streets.node[u_node]['y']
            coord_x2 = streets.node[v_node]['x']
            coord_y2 = streets.node[v_node]['y']
            data['geometry'] = geom.LineString(
                [(coord_x1, coord_y1), (coord_x2, coord_y2)])


def check_geometry(streets):
    """ Checks if all edges of the graph have a geometry object"""
    complete = True
    for _, _, data in streets.edges(data=True):
        if 'geometry' not in data:
            complete = False
            break

    return complete

def line_route_between_nodes(node_from, node_to, graph):
    """Determines the line representing the shortest path between two nodes"""

    route = nx.shortest_path(graph, node_from, node_to, weight='length')
    edge_nodes = list(zip(route[:-1], route[1:]))
    # TODO: preallocate?
    lines = []
    for u_node, v_node in edge_nodes:
        # If there are parallel edges, select the shortest in length
        data = min([data for data in graph.edge[u_node][v_node].values()], \
                   key=lambda x: x['length'])
        lines.append(data['geometry'])

    line = ops.linemerge(lines)
    return line
