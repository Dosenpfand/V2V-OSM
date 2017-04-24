""" Determines the propagation conditions (LOS/OLOS/NLOS orthogonal/NLOS paralell) of connections"""

import numpy as np
import shapely.geometry as geom
import osmnx_addons
import geometry as geom_o
import networkx as nx
from networkx.algorithms.approximation.connectivity import \
    local_node_connectivity as nx_local_node_connectivity
from networkx.algorithms.connectivity import \
    local_edge_connectivity as nx_local_edge_connectivity
import utils


def path_redundancy(graph, node, distances):
    """Determines the path redundancy (number of node/edge disjoint paths) 
    from one specific node to all other nodes"""
    # NOTE: we calculate the minimum number of node independent paths as an approximation (and not
    # the maximum)

    count_nodes = graph.number_of_nodes()
    path_redundancy = np.zeros(
        count_nodes - 1,
        dtype=[('distance', 'float'),
               ('count_node_disjoint_paths', 'uint'),
               ('count_edge_disjoint_paths', 'uint')])
    iter_veh = 0
    for node_iter_veh in graph.nodes():
        if node_iter_veh == node:
            continue
        idx_cond = utils.square2cond(
            count_nodes, node, node_iter_veh)
        path_redundancy[iter_veh]['distance'] = distances[idx_cond]

        path_redundancy[iter_veh]['count_node_disjoint_paths'] = nx_local_node_connectivity(
            graph, source=node, target=node_iter_veh)
        path_redundancy[iter_veh]['count_edge_disjoint_paths'] = nx_local_edge_connectivity(
            graph, node, node_iter_veh)
        iter_veh += 1

    return path_redundancy


def veh_cons_are_nlos(point_own, points_vehs, buildings, max_dist=None):
    """ Determines for each connection if it is NLOS or not (i.e. LOS and OLOS)"""

    is_nlos = np.ones(np.size(points_vehs), dtype=bool)

    for index, point in enumerate(points_vehs):
        line = geom.LineString([point_own, point])
        if (max_dist is None) or (line.length < max_dist):
            is_nlos[index] = geom_o.line_intersects_buildings(line, buildings)

    return is_nlos


def veh_cons_are_nlos_all(points_vehs, buildings, max_dist=None):
    """ Determines for each possible connection if it is NLOS or not (i.e. LOS and OLOS)"""

    count_vehs = np.size(points_vehs)
    count_cond = count_vehs * (count_vehs - 1) // 2
    is_nlos = np.ones(count_cond, dtype=bool)

    index = 0
    for idx1, point1 in enumerate(points_vehs):
        for point2 in points_vehs[idx1 + 1:]:
            line = geom.LineString([point1, point2])
            if (max_dist is None) or (line.length < max_dist):
                is_nlos[index] = geom_o.line_intersects_buildings(
                    line, buildings)
            index += 1

    return is_nlos


def veh_cons_are_olos(point_own, point_vehs, margin=1):
    """ Determines for each LOS/OLOS connection if it is OLOS """

    # TODO: Also use NLOS vehicles!
    # TODO: working properly? still too many LOS vehicles?

    is_olos = np.zeros(np.size(point_vehs), dtype=bool)

    for index, point in np.ndenumerate(point_vehs):
        line = geom.LineString([point_own, point])
        indices_other = np.ones(np.size(point_vehs), dtype=bool)
        indices_other[index] = False
        is_olos[index] = geom_o.line_intersects_points(line, point_vehs[indices_other],
                                                       margin=margin)

    return is_olos


def check_if_cons_orthogonal(streets_wave, graph_veh_own, graphs_veh_other, max_angle=np.pi):
    """Determines if the condition is NLOS on an orthogonal street for every possible connection to
    one node """
    node_own = graph_veh_own.graph['node_veh']
    streets_wave_local = nx.compose(graph_veh_own, streets_wave)
    count_veh_other = np.size(graphs_veh_other)

    is_orthogonal = np.zeros(count_veh_other, dtype=bool)
    coords_max_angle = np.zeros((count_veh_other, 2))
    for index, graph in enumerate(graphs_veh_other):

        node_v = graph.graph['node_veh']
        streets_wave_local_iter = nx.compose(graph, streets_wave_local)

        # TODO: Use angles as weight and not length?
        route = osmnx_addons.line_route_between_nodes(
            node_own, node_v, streets_wave_local_iter)
        angles = geom_o.angles_along_line(route)
        angles_wrapped = np.pi - np.abs(geom_o.wrap_to_pi(angles))

        sum_angles = sum(angles_wrapped)
        if sum_angles < max_angle:
            is_orthogonal[index] = True
        else:
            is_orthogonal[index] = False

        # Determine position of max angle
        index_angle = np.argmax(angles_wrapped)
        route_coords = np.array(route.xy)
        coords_max_angle[index, :] = route_coords[:, index_angle + 1]

    return is_orthogonal, coords_max_angle


def add_edges_if_los(graph, buildings, max_distance=50):
    """Adds edges to the streets graph if there is none between 2 nodes if there is none, the have
    no buildings in between and are only a certain distance apart"""

    for index, node_u in enumerate(graph.nodes()):
        coords_u = np.array((graph.node[node_u]['x'], graph.node[node_u]['y']))
        for node_v in graph.nodes()[index + 1:]:

            # Check if nodes are already connected
            if graph.has_edge(node_u, node_v):
                continue
            coords_v = np.array(
                (graph.node[node_v]['x'], graph.node[node_v]['y']))
            distance = np.linalg.norm(coords_u - coords_v, ord=2)

            # Check if the nodes are further apart than the max distance
            if distance > max_distance:
                continue

            # Check if there are buildings between the nodes
            line = geom.LineString(
                [(coords_u[0], coords_u[1]), (coords_v[0], coords_v[1])])

            if geom_o.line_intersects_buildings(line, buildings):
                continue

            # Add edge between nodes
            edge_attr = {'length': distance, 'geometry': line}
            graph.add_edge(node_u, node_v, attr_dict=edge_attr)
