""" Determines the propagation conditions (LOS/OLOS/NLOS orthogonal/NLOS paralell) of connections"""

from enum import IntEnum
import numpy as np
import shapely.geometry as geom
import osmnx_addons
import geometry as geom_o
import networkx as nx


class Cond(IntEnum):
    """Enumeration of possible propagation conditions"""

    LOS = 1
    OLOS = 2
    NLOS_par = 3
    NLOS_ort = 4
    OLOS_LOS = 5
    NLOS = 6


def gen_prop_cond_matrix(points_vehs,
                         buildings,
                         graph_streets_wave=None,
                         graphs_vehs=None,
                         fully_determine=True,
                         max_dist=None,
                         car_radius=2,
                         max_angle=np.pi):
    """Determines the condensed connection matrix, i.e. the propagation conditions between all pairs
    of vehicles"""

    count_vehs = points_vehs.size
    count_cond = count_vehs * (count_vehs - 1) // 2
    prop_cond_matrix = np.zeros(count_cond, dtype=Cond)
    coords_max_angle_matrix = np.zeros(count_cond, dtype=object)
    range_vehs = np.arange(count_vehs)

    index = 0
    for idx1, point1 in enumerate(points_vehs):
        for idx2, point2 in enumerate(points_vehs[idx1 + 1:]):
            is_nlos = True
            line = geom.LineString([point1, point2])
            if (max_dist is None) or (line.length < max_dist):
                is_nlos = geom_o.line_intersects_buildings(
                    line, buildings)

            if is_nlos:
                if fully_determine:
                    graph_veh1 = graphs_vehs[idx1]
                    graph_veh2 = graphs_vehs[idx1 + idx2 + 1]

                    is_orthogonal, coords_max_angle = check_if_con_is_orthogonal(
                        graph_streets_wave,
                        graph_veh1,
                        graph_veh2,
                        max_angle=max_angle)
                    if is_orthogonal:
                        prop_cond_matrix[index] = Cond.NLOS_ort
                        coords_max_angle_matrix[index] = coords_max_angle
                    else:
                        prop_cond_matrix[index] = Cond.NLOS_par

                else:
                    prop_cond_matrix[index] = Cond.NLOS
            else:
                if fully_determine:
                    idxs_other = np.setdiff1d(
                        range_vehs, [idx1, idx1 + idx2 + 1])
                    is_olos = geom_o.line_intersects_points(line, points_vehs[idxs_other],
                                                            margin=car_radius)
                    if is_olos:
                        prop_cond_matrix[index] = Cond.OLOS
                    else:
                        prop_cond_matrix[index] = Cond.LOS
                else:
                    prop_cond_matrix[index] = Cond.OLOS_LOS

            index += 1

    return prop_cond_matrix, coords_max_angle_matrix


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
    # TODO: delete this function? completele replacable by
    # gen_prop_cond_matrix?

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


def veh_cons_are_olos(point_own, points_vehs, margin=1):
    """Determines for each LOS/OLOS connection if it is OLOS"""

    is_olos = np.zeros(np.size(points_vehs), dtype=bool)

    for index, point in np.ndenumerate(points_vehs):
        line = geom.LineString([point_own, point])
        indices_other = np.ones(np.size(points_vehs), dtype=bool)
        indices_other[index] = False
        is_olos[index] = geom_o.line_intersects_points(line, points_vehs[indices_other],
                                                       margin=margin)

    return is_olos


def check_if_con_is_orthogonal(streets_wave,
                               graph_veh_u,
                               graph_veh_v,
                               max_angle=np.pi):
    """Determines if the propagation condition between two vehicles is NLOS on an orthogonal
    street"""

    node_u = graph_veh_u.graph['node_veh']
    node_v = graph_veh_v.graph['node_veh']
    streets_wave_local = nx.compose(graph_veh_u, streets_wave)
    streets_wave_local = nx.compose(graph_veh_v, streets_wave_local)

    # TODO: Use angles as weight and not length?
    route = osmnx_addons.line_route_between_nodes(
        node_u, node_v, streets_wave_local)
    angles = geom_o.angles_along_line(route)
    angles_wrapped = np.pi - np.abs(geom_o.wrap_to_pi(angles))

    sum_angles = sum(angles_wrapped)
    if sum_angles <= max_angle:
        is_orthogonal = True
    else:
        is_orthogonal = False

    # Determine position of max angle
    index_angle = np.argmax(angles_wrapped)
    route_coords = np.array(route.xy)
    coords_max_angle = route_coords[:, index_angle + 1]

    return is_orthogonal, coords_max_angle


def check_if_cons_are_orthogonal(streets_wave,
                                 graph_veh_own,
                                 graphs_veh_other,
                                 max_angle=np.pi):
    """Determines if the propagation condition is NLOS on an orthogonal street for every possible
    connection to one node"""

    count_veh_other = np.size(graphs_veh_other)

    is_orthogonal = np.zeros(count_veh_other, dtype=bool)
    coords_max_angle = np.zeros((count_veh_other, 2))
    for index, graph in enumerate(graphs_veh_other):

        is_orthogonal[index], coords_max_angle[index, :] = \
            check_if_con_is_orthogonal(
                streets_wave, graph_veh_own, graph, max_angle=max_angle)

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
