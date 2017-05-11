""" Provides functions to generate connection graphs and matrices and
derive further results from them"""

import logging
import utils
import networkx as nx
import numpy as np
import geometry as geom_o
import scipy.spatial.distance as dist
import propagation as prop
from networkx.algorithms.approximation.connectivity import \
    local_node_connectivity as nx_local_node_connectivity
from networkx.algorithms.connectivity import \
    local_edge_connectivity as nx_local_edge_connectivity



def gen_connection_matrix(vehs, gdf_buildings, max_metric, metric='distance'):
    """Simulates links between every set of 2 vehicles and determines if they are connected using
    either distance or pathloss as a metric. Returns a matrix"""

    # Initialize
    if metric not in ['distance', 'pathloss']:
        raise NotImplementedError('Metric not supported')

    if metric == 'distance':
        if isinstance(max_metric, dict):
            max_dist_nlos = max_metric['nlos']
            max_dist_olos_los = max_metric['olos_los']
        else:
            max_dist_nlos = max_metric
            max_dist_olos_los = max_metric
        max_dist = max(max_dist_nlos, max_dist_olos_los)
    elif metric == 'pathloss':
        # TODO: set a very high max distance?
        max_dist = None
    else:
        raise NotImplementedError('Metric not implemented')

    count_veh = vehs.count
    count_cond = count_veh * (count_veh - 1) // 2
    vehs.allocate(count_cond)

    # Determine NLOS and OLOS/LOS
    time_start = utils.debug(None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos_all(
        vehs.get_points(), gdf_buildings, max_dist=max_dist)
    is_olos_los = np.invert(is_nlos)
    idxs_nlos = np.where(is_nlos)[0]
    idxs_olos_los = np.where(is_olos_los)[0]
    vehs.add_key('nlos', idxs_nlos)
    vehs.add_key('olos_los', idxs_olos_los)

    count_cond = count_veh * (count_veh - 1) // 2

    utils.debug(time_start)

    # Determine in range vehicles
    time_start = utils.debug(None, 'Determining in range vehicles')

    distances = dist.pdist(vehs.coordinates)
    if metric == 'distance':
        idxs_in_range_olos_los = idxs_olos_los[
            distances[idxs_olos_los] < max_dist_olos_los]
        idxs_in_range_nlos = idxs_nlos[
            distances[idxs_nlos] < max_dist_nlos]
        idxs_in_range = np.append(
            idxs_in_range_olos_los, idxs_in_range_nlos)
        idxs_out_range = np.setdiff1d(np.arange(count_cond), idxs_in_range)
        vehs.add_key('in_range', idxs_in_range)
        vehs.add_key('out_range', idxs_out_range)
    elif metric == 'pathloss':
        # TODO: !
        raise NotImplementedError('TODO!')
    else:
        raise NotImplementedError('Metric not implemented')

    is_in_range = np.in1d(np.arange(count_cond), idxs_in_range)
    matrix_cons = dist.squareform(is_in_range).astype(bool)

    return matrix_cons


def gen_connection_graph(vehs, gdf_buildings, max_metric, metric='distance'):
    """Simulates links between every set of 2 vehicles and determines if they are connected using
    either distance or pathloss as a metric. Returns a networkx graph"""

    matrix_cons = gen_connection_matrix(vehs,
                                        gdf_buildings,
                                        max_metric,
                                        metric=metric)

    # TODO: check if node names correspond to same indices as in vehs?
    graph_cons = nx.from_numpy_matrix(matrix_cons)

    return graph_cons


def calc_net_connectivity(graph_cons, vehs=None):
    """Calculates the network connectivity (relative size of the biggest connected cluster)"""

    # Find biggest cluster
    time_start = utils.debug(None, 'Finding biggest cluster')
    count_veh = graph_cons.order()
    clusters = nx.connected_component_subgraphs(graph_cons)
    cluster_max = max(clusters, key=len)
    net_connectivity = cluster_max.order() / count_veh
    if vehs is not None:
        vehs.add_key('cluster_max', cluster_max.nodes())
        not_cluster_max_nodes = np.arange(count_veh)[~np.in1d(
            np.arange(count_veh), cluster_max.nodes())]
        vehs.add_key('not_cluster_max', not_cluster_max_nodes)

    utils.debug(time_start)
    logging.info('Network connectivity {:.2f}%'.format(
        net_connectivity * 100))

    return net_connectivity


def calc_path_redundancy(graph_cons, vehs):
    """Calculates the path redundancy of the connection graph for the center vehicle"""

    # Find center vehicle
    count_veh = vehs.count
    time_start = utils.debug(None, 'Finding center vehicle')
    idx_center_veh = geom_o.find_center_veh(vehs.get())
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    utils.debug(time_start)

    # Determine path redundancy
    # NOTE: we calculate the minimum number of node independent paths as an approximation (and not
    # the maximum)
    node_center_veh = idx_center_veh  # TODO: this does not seem to be the center?
    time_start = utils.debug(None, 'Determining path redundancy')
    distances = dist.pdist(vehs.coordinates)
    path_redundancy = get_path_redundancy(
        graph_cons, node_center_veh, distances)

    utils.debug(time_start)

    return path_redundancy


def get_path_redundancy(graph, node, distances):
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
