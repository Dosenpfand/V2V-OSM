""" Provides functions to generate connection graphs and matrices and
derive further results from them"""

import logging

import networkx as nx
import numpy as np
import scipy.spatial.distance as dist
from networkx.algorithms.approximation.connectivity import \
    local_node_connectivity as nx_local_node_connectivity
from networkx.algorithms.connectivity import \
    local_edge_connectivity as nx_local_edge_connectivity

import geometry as geom_o
import pathloss
import propagation as prop
import utils


def gen_connection_matrix(vehs,
                          gdf_buildings,
                          max_metric,
                          metric='distance',
                          graph_streets_wave=None,
                          metric_config=None):
    """Simulates links between every set of 2 vehicles and determines if they are connected using
    either distance or pathloss as a metric. Returns a matrix"""

    # Initialize
    if metric not in ['distance', 'pathloss']:
        raise NotImplementedError('Metric not supported')

    count_veh = vehs.count
    count_cond = count_veh * (count_veh - 1) // 2
    vehs.allocate(count_cond)

    if metric == 'distance':
        if isinstance(max_metric, dict):
            max_dist_nlos = max_metric['nlos']
            max_dist_olos_los = max_metric['olos_los']
        else:
            max_dist_nlos = max_metric
            max_dist_olos_los = max_metric

        max_dist = max(max_dist_nlos, max_dist_olos_los)

        # Determine NLOS and OLOS/LOS
        time_start = utils.debug(None, 'Determining propagation conditions')

        # Determine propagation condition matrix
        prop_cond_matrix, _ = prop.gen_prop_cond_matrix(
            vehs.get_points(),
            gdf_buildings,
            graph_streets_wave=None,
            graphs_vehs=None,
            fully_determine=False,
            max_dist=max_dist)

        idxs_olos_los = np.nonzero(prop_cond_matrix == prop.Cond.OLOS_LOS)[0]
        idxs_nlos = np.nonzero(prop_cond_matrix == prop.Cond.NLOS)[0]
        vehs.add_key('nlos', idxs_nlos)
        vehs.add_key('olos_los', idxs_olos_los)

        count_cond = count_veh * (count_veh - 1) // 2

        utils.debug(time_start)

        # Determine in range vehicles
        time_start = utils.debug(None, 'Determining in range vehicles')

        distances = dist.pdist(vehs.coordinates)

        idxs_in_range_olos_los = idxs_olos_los[
            distances[idxs_olos_los] < max_dist_olos_los]
        idxs_in_range_nlos = idxs_nlos[
            distances[idxs_nlos] < max_dist_nlos]
        idxs_in_range = np.append(
            idxs_in_range_olos_los, idxs_in_range_nlos)
        idxs_out_range = np.setdiff1d(np.arange(count_cond), idxs_in_range)
        vehs.add_key('in_range', idxs_in_range)
        vehs.add_key('out_range', idxs_out_range)
        utils.debug(time_start)

    elif metric == 'pathloss':
        if graph_streets_wave is None:
            raise RuntimeError('Streets wave propagation graph not given')

        if metric_config is None:
            metric_config = {}
        if 'shadowfading_enabled' not in metric_config:
            metric_config['shadowfading_enabled'] = True
        if 'max_dist' not in metric_config:
            metric_config['max_dist'] = None
        if 'car_radius' not in metric_config:
            metric_config['car_radius'] = 1.5
        if 'max_angle' not in metric_config:
            metric_config['max_angle'] = np.pi

        # Determine propagation condition matrix
        prop_cond_matrix, coords_max_angle_matrix = prop.gen_prop_cond_matrix(
            vehs.get_points(),
            gdf_buildings,
            graph_streets_wave=graph_streets_wave,
            graphs_vehs=vehs.get_graph(),
            fully_determine=True,
            max_dist=metric_config['max_dist'],
            car_radius=metric_config['car_radius'],
            max_angle=metric_config['max_angle'])

        idxs_los = np.nonzero(prop_cond_matrix == prop.Cond.LOS)[0]
        idxs_olos = np.nonzero(prop_cond_matrix == prop.Cond.OLOS)[0]
        idxs_nlos_ort = np.nonzero(prop_cond_matrix == prop.Cond.NLOS_ort)[0]
        idxs_nlos_par = np.nonzero(prop_cond_matrix == prop.Cond.NLOS_par)[0]
        idxs_olos_los = np.sort(np.append(idxs_los, idxs_olos))
        idxs_nlos = np.sort(np.append(idxs_nlos_ort, idxs_nlos_par))

        vehs.add_key('los', idxs_los)
        vehs.add_key('olos', idxs_olos)
        vehs.add_key('nlos_ort', idxs_nlos_ort)
        vehs.add_key('nlos_par', idxs_nlos_par)
        vehs.add_key('olos_los', idxs_olos_los)
        vehs.add_key('nlos', idxs_nlos)

        distances = dist.pdist(vehs.coordinates)

        ploss = pathloss.Pathloss()
        if not metric_config['shadowfading_enabled']:
            ploss.disable_shadowfading()

        pathlosses = np.zeros(distances.size)
        pathlosses[idxs_los] = ploss.pathloss_los(distances[idxs_los])
        pathlosses[idxs_olos] = ploss.pathloss_olos(distances[idxs_olos])
        pathlosses[idxs_nlos_par] = np.inf

        vehs_coords = vehs.get()
        for idx_nlos_ort in idxs_nlos_ort:
            coords_max_angle = coords_max_angle_matrix[idx_nlos_ort]
            idx_veh1, idx_veh2 = utils.condensed_to_square(
                idx_nlos_ort, count_veh)
            coords_veh1 = vehs_coords[idx_veh1]
            coords_veh2 = vehs_coords[idx_veh2]

            dist_1 = np.linalg.norm(coords_veh1 - coords_max_angle)
            dist_2 = np.linalg.norm(coords_veh2 - coords_max_angle)
            pathloss_iter1 = ploss.pathloss_nlos(dist_1, dist_2)
            pathloss_iter2 = ploss.pathloss_nlos(dist_2, dist_1)


            # NOTE: Maximum of the 2 pathlosses => Nodes connected if both are below threshold
            pathlosses[idx_nlos_ort] = np.max(
                [pathloss_iter1, pathloss_iter2])

        idxs_in_range = np.nonzero(pathlosses < max_metric)
        idxs_out_range = np.setdiff1d(np.arange(count_cond), idxs_in_range)
        vehs.add_key('in_range', idxs_in_range)
        vehs.add_key('out_range', idxs_out_range)

    else:
        raise NotImplementedError('Metric not implemented')

    is_in_range = np.in1d(np.arange(count_cond), idxs_in_range)
    matrix_cons = dist.squareform(is_in_range).astype(bool)

    return matrix_cons


def gen_connection_graph(vehs,
                         gdf_buildings,
                         max_metric,
                         metric='distance',
                         graph_streets_wave=None,
                         metric_config=None):
    """Simulates links between every set of 2 vehicles and determines if they are connected using
    either distance or pathloss as a metric. Returns a networkx graph"""

    matrix_cons = gen_connection_matrix(vehs,
                                        gdf_buildings,
                                        max_metric,
                                        metric=metric,
                                        graph_streets_wave=graph_streets_wave,
                                        metric_config=metric_config
                                        )

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
    node_center_veh = idx_center_veh
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
        idx_cond = utils.square_to_condensed(
            node, node_iter_veh, count_nodes)
        path_redundancy[iter_veh]['distance'] = distances[idx_cond]

        path_redundancy[iter_veh]['count_node_disjoint_paths'] = nx_local_node_connectivity(
            graph, source=node, target=node_iter_veh)
        path_redundancy[iter_veh]['count_edge_disjoint_paths'] = nx_local_edge_connectivity(
            graph, node, node_iter_veh)
        iter_veh += 1

    return path_redundancy
