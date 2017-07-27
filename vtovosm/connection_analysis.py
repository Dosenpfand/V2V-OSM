""" Provides functions to generate connection graphs and matrices and
derive further results from them"""

import multiprocessing as mp
from collections import namedtuple

import networkx as nx
import numpy as np
import scipy.spatial.distance as sp_dist
import networkx.algorithms.connectivity as nx_con
import networkx.algorithms.approximation.connectivity as nx_con_approx
from . import geometry as geom_o
from . import pathloss
from . import propagation as prop
from . import utils


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

        distances = sp_dist.pdist(vehs.coordinates)

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

        distances = sp_dist.pdist(vehs.coordinates)

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
    matrix_cons = sp_dist.squareform(is_in_range).astype(bool)

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


def calc_net_connectivities(graphs_cons):
    """Calculates the network connectivities (relative size of the biggest connected cluster)"""

    net_connectivities = np.zeros(len(graphs_cons), dtype=object)

    for idx, graph_cons in enumerate(graphs_cons):
        net_connectivities[idx] = calc_net_connectivity(graph_cons)

    return net_connectivities


NetworkConnectivity = namedtuple('NetworkConnectivity', ['net_connectivity', 'min_node_cut', 'count_cluster'])


def calc_net_connectivity(graph_cons, vehs=None, cut_only_fully_connected=True):
    """Calculates the network connectivity (relative size of the biggest connected cluster)"""

    time_start = utils.debug(None, 'Finding biggest cluster')

    # Find biggest cluster
    count_veh = graph_cons.order()
    clusters = list(nx.connected_component_subgraphs(graph_cons))
    cluster_max = max(clusters, key=len)
    count_veh_max = cluster_max.order()
    net_connectivity = count_veh_max / count_veh
    if vehs is not None:
        cluster_nodes = cluster_max.nodes()
        vehs.add_key('cluster_max', cluster_nodes)
        not_cluster_max_nodes = np.arange(count_veh)[~np.in1d(
            np.arange(count_veh), cluster_nodes)]
        vehs.add_key('not_cluster_max', not_cluster_max_nodes)

    # Find the minimum node cut
    count_cluster = len(clusters)
    if count_cluster == 1 or not cut_only_fully_connected:
        min_node_cut = nx.minimum_node_cut(cluster_max)
    else:
        min_node_cut = None

    result = NetworkConnectivity(net_connectivity=net_connectivity,
                                 min_node_cut=min_node_cut,
                                 count_cluster=count_cluster)
    utils.debug(time_start)

    return result


def calc_center_path_redundancies(graphs_cons, vehs):
    """Determines the path redundancy of all the connection graphs for the center vehicle"""

    count_graphs = len(graphs_cons)
    count_nodes = graphs_cons[0].number_of_nodes()

    path_redundancies = np.zeros(
        (count_nodes - 1, count_graphs),
        dtype=[('distance', 'float'),
               ('count_node_disjoint_paths', 'uint'),
               ('count_edge_disjoint_paths', 'uint')])

    for idx, graph_cons, vehs_snapshot in zip(range(count_graphs), graphs_cons, vehs):
        path_redundancy = calc_center_path_redundancy(graph_cons, vehs_snapshot)
        path_redundancies[:, idx] = path_redundancy

    return path_redundancies


def calc_center_path_redundancy(graph_cons, vehs):
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
    distances = sp_dist.pdist(vehs.coordinates)
    path_redundancy = calc_path_redundancy(
        graph_cons, node_center_veh, distances)

    utils.debug(time_start)

    return path_redundancy


def calc_path_redundancies(graph, vehs):
    """Determines the path redundancies (number of node disjoint paths) for all pairs of nodes."""
    # NOTE: we calculate the minimum number of node independent paths as an approximation (and not
    # the maximum)

    time_start = utils.debug(None, 'Determining path redundancies')

    distances = sp_dist.pdist(vehs.coordinates)
    count_nodes = graph.number_of_nodes()
    node_cons = nx_con_approx.all_pairs_node_connectivity(graph)
    node_cons_dist = {}

    for u, vals in node_cons.items():
        node_cons_dist[u] = {}
        for v, val in vals.items():
            idx_cond = utils.square_to_condensed(u, v, count_nodes)
            node_cons_dist[u][v] = {'node_con': val, 'dist': distances[idx_cond]}

    utils.debug(time_start)

    return node_cons_dist


def calc_path_redundancy(graph, node, distances):
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

        path_redundancy[iter_veh]['count_node_disjoint_paths'] = nx_con_approx.local_node_connectivity(
            graph, source=node, target=node_iter_veh)
        path_redundancy[iter_veh]['count_edge_disjoint_paths'] = nx_con.local_edge_connectivity(
            graph, node, node_iter_veh)
        iter_veh += 1

    return path_redundancy


LinkDurations = namedtuple('LinkDurations',
                           ['durations_con', 'durations_discon', 'durations_matrix_con', 'durations_matrix_discon'])


def calc_link_durations(graphs_cons):
    """Determines the link durations (continuous time period during which 2 nodes are directly connected)
    and link rehealing time (lengths of disconnected periods)"""

    # Assumes that all graphs have the same number of nodes
    count_nodes = graphs_cons[0].number_of_nodes()
    size_cond = count_nodes * (count_nodes - 1) // 2

    durations_matrix_con = np.zeros(size_cond, dtype=object)
    for idx in range(durations_matrix_con.size):
        durations_matrix_con[idx] = []

    durations_matrix_discon = np.zeros(size_cond, dtype=object)
    for idx in range(durations_matrix_discon.size):
        durations_matrix_discon[idx] = []

    active_matrix = np.zeros(size_cond, bool)

    for graph_cons in graphs_cons:

        # Search for all active connections
        edges = [set(edge) for edge in graph_cons.edges_iter()]

        for idx_u, node_u in enumerate(graph_cons.nodes()):
            for node_v in graph_cons.nodes()[idx_u + 1:]:
                idx_cond = utils.square_to_condensed(node_u, node_v, count_nodes)
                is_connected = {node_u, node_v} in edges
                if is_connected:
                    if not active_matrix[idx_cond]:
                        durations_matrix_con[idx_cond].append(0)
                        active_matrix[idx_cond] = True
                    durations_matrix_con[idx_cond][-1] += 1
                else:
                    if active_matrix[idx_cond] or durations_matrix_discon[idx_cond] == []:
                        durations_matrix_discon[idx_cond].append(0)
                        active_matrix[idx_cond] = False
                    durations_matrix_discon[idx_cond][-1] += 1

    durations_con = [item for sublist in durations_matrix_con.tolist() for item in sublist]
    durations_discon = [item for sublist in durations_matrix_discon.tolist() for item in sublist]

    link_durations = LinkDurations(durations_con=durations_con,
                                   durations_discon=durations_discon,
                                   durations_matrix_con=durations_matrix_con,
                                   durations_matrix_discon=durations_matrix_discon)

    return link_durations


def calc_link_durations_multiprocess(graphs_cons, processes=None, chunk_length=None):
    """Determines the link durations using multiple processes. See also: calc_link_durations"""

    # TODO: optionally return 1st and last con/discon matrix to speed up merge in multiprocessing case? also applies to
    # calc_connection_durations_multiprocess

    # Process chunks in parallel
    with mp.Pool(processes=processes) as pool:
        # Determine optimal chunk size
        if chunk_length is None:
            # NOTE: _processes() should not be used, but no obvious alternative
            # TODO: maybe use smaller chunks to use less memory at any given time? also applies to
            # calc_connection_durations_multiprocess
            chunk_length = int(np.ceil(len(graphs_cons) / pool._processes))

        # Split the graphs list in chunks
        graphs_chunks = [graphs_cons[i:i + chunk_length] for i in range(0, len(graphs_cons), chunk_length)]

        # Run parallel calculation
        link_chunks = pool.map(calc_link_durations, graphs_chunks)

    # Merge link durations
    durations_merged = merge_link_durations(link_chunks, graphs_cons, chunk_length)

    return durations_merged


ConnectionDurations = namedtuple('ConnectionDurations',
                                 ['durations_con', 'durations_discon', 'durations_matrix_con',
                                  'durations_matrix_discon'])


def calc_connection_durations(graphs_cons):
    """Determines the connection durations (continuous time period during which 2 nodes have a path between them)
    and rehealing times (lengths of disconnected periods)"""

    # Assumes that all graphs have the same number of nodes
    count_nodes = graphs_cons[0].number_of_nodes()
    size_cond = count_nodes * (count_nodes - 1) // 2

    durations_matrix_con = np.zeros(size_cond, dtype=object)
    for idx in range(durations_matrix_con.size):
        durations_matrix_con[idx] = []

    durations_matrix_discon = np.zeros(size_cond, dtype=object)
    for idx in range(durations_matrix_discon.size):
        durations_matrix_discon[idx] = []

    active_matrix = np.zeros(size_cond, bool)

    for graph_cons in graphs_cons:

        # Determine all nodes that have a path between them
        has_path_matrix = to_has_path_matrix(graph_cons)

        # Search for all active connections
        connections = []
        for idx_u, node_u in enumerate(graph_cons.nodes()):
            for node_v in graph_cons.nodes()[idx_u + 1:]:
                idx_cond = utils.square_to_condensed(node_u, node_v, count_nodes)
                is_connected = has_path_matrix[idx_cond]
                if is_connected:
                    if not active_matrix[idx_cond]:
                        durations_matrix_con[idx_cond].append(0)
                        active_matrix[idx_cond] = True
                    durations_matrix_con[idx_cond][-1] += 1
                    connections.append((node_u, node_v))
                else:
                    if active_matrix[idx_cond] or durations_matrix_discon[idx_cond] == []:
                        durations_matrix_discon[idx_cond].append(0)
                        active_matrix[idx_cond] = False
                    durations_matrix_discon[idx_cond][-1] += 1

    durations_con = [item for sublist in durations_matrix_con.tolist() for item in sublist]
    durations_discon = [item for sublist in durations_matrix_discon.tolist() for item in sublist]

    connection_durations = ConnectionDurations(durations_con=durations_con,
                                               durations_discon=durations_discon,
                                               durations_matrix_con=durations_matrix_con,
                                               durations_matrix_discon=durations_matrix_discon)

    return connection_durations


def to_has_path_matrix(graph):
    """Determine all nodes that have a path between them"""

    count_nodes = graph.number_of_nodes()
    size_cond = count_nodes * (count_nodes - 1) // 2
    has_path_matrix = np.zeros(size_cond, dtype=bool)

    # This is much faster than calling nx.has_path() for every source and destination node
    path_lengths = nx.all_pairs_shortest_path_length(graph)

    for node_u, nodes_v in path_lengths.items():
        for node_v in nodes_v.keys():
            if node_u <= node_v:
                continue
            idx_cond = utils.square_to_condensed(node_v, node_u, count_nodes)
            has_path_matrix[idx_cond] = True

    return has_path_matrix


def calc_connection_durations_multiprocess(graphs_cons, processes=None, chunk_length=None):
    """Determines the connection durations using multiple processes. See also: calc_connection_durations"""

    # Process chunks in parallel
    with mp.Pool(processes=processes) as pool:
        # Determine optimal chunk size
        if chunk_length is None:
            # NOTE: _processes() should not be used, but no obvious alternative
            chunk_length = int(np.ceil(len(graphs_cons) / pool._processes))

        # Split the graphs list in chunks
        graphs_chunks = [graphs_cons[i:i + chunk_length] for i in range(0, len(graphs_cons), chunk_length)]

        # Run parallel calculation
        connection_chunks = pool.map(calc_connection_durations, graphs_chunks)

    # Merge link durations
    durations_merged = merge_connection_durations(connection_chunks, graphs_cons, chunk_length)

    return durations_merged


def merge_link_durations(link_chunks, graphs_cons, chunk_length):
    """Merges the link duration chunks from multiprocessing"""

    size_cond = link_chunks[0].durations_matrix_con.size
    durations_matrix_con = np.zeros(size_cond, dtype=object)
    durations_matrix_discon = np.zeros(size_cond, dtype=object)

    for idx_link in range(size_cond):
        link_pair_durations_con = link_chunks[0].durations_matrix_con[idx_link]
        link_pair_durations_discon = link_chunks[0].durations_matrix_discon[idx_link]
        idx_square1, idx_square2 = utils.condensed_to_square(idx_link, graphs_cons[0].number_of_nodes())
        for idx_chunk in range(1, len(link_chunks)):
            link_pair_durations_con_iter = link_chunks[idx_chunk].durations_matrix_con[idx_link]
            link_pair_durations_discon_iter = link_chunks[idx_chunk].durations_matrix_discon[idx_link]
            idx_graph = chunk_length * idx_chunk

            has_edge_1 = graphs_cons[idx_graph].has_edge(idx_square1, idx_square2)
            has_edge_2 = graphs_cons[idx_graph - 1].has_edge(idx_square1, idx_square2)
            to_merge_con = has_edge_1 and has_edge_2
            to_merge_discon = not (has_edge_1 or has_edge_2)

            if to_merge_con:
                link_pair_durations_con[-1] += link_pair_durations_con_iter[0]
                link_pair_durations_con_iter.pop(0)

            if to_merge_discon:
                link_pair_durations_discon[-1] += link_pair_durations_discon_iter[0]
                link_pair_durations_discon_iter.pop(0)

            link_pair_durations_con += link_pair_durations_con_iter
            link_pair_durations_discon += link_pair_durations_discon_iter

        durations_matrix_con[idx_link] = link_pair_durations_con
        durations_matrix_discon[idx_link] = link_pair_durations_discon

    durations_con = [item for sublist in durations_matrix_con.tolist() for item in sublist]
    durations_discon = [item for sublist in durations_matrix_discon.tolist() for item in sublist]

    link_durations = LinkDurations(durations_con=durations_con,
                                   durations_discon=durations_discon,
                                   durations_matrix_con=durations_matrix_con,
                                   durations_matrix_discon=durations_matrix_discon)

    return link_durations


def merge_connection_durations(connection_chunks, graphs_cons, chunk_length):
    """Merges the connection duration chunks from multiprocessing"""

    size_cond = connection_chunks[0].durations_matrix_con.size
    durations_matrix_con = np.zeros(size_cond, dtype=object)
    durations_matrix_discon = np.zeros(size_cond, dtype=object)

    for idx_link in range(size_cond):
        connection_pair_durations_con = connection_chunks[0].durations_matrix_con[idx_link]
        connection_pair_durations_discon = connection_chunks[0].durations_matrix_discon[idx_link]
        idx_square1, idx_square2 = utils.condensed_to_square(idx_link, graphs_cons[0].number_of_nodes())
        for idx_chunk in range(1, len(connection_chunks)):
            connection_pair_durations_con_iter = connection_chunks[idx_chunk].durations_matrix_con[idx_link]
            connection_pair_durations_discon_iter = connection_chunks[idx_chunk].durations_matrix_discon[idx_link]
            idx_graph = chunk_length * idx_chunk

            # TODO: change following 2 lines to use to_has_path_matrix
            has_path_1 = nx.has_path(graphs_cons[idx_graph], idx_square1, idx_square2)
            has_path_2 = nx.has_path(graphs_cons[idx_graph - 1], idx_square1, idx_square2)

            to_merge_con = has_path_1 and has_path_2
            to_merge_discon = not (has_path_1 or has_path_2)

            if to_merge_con:
                connection_pair_durations_con[-1] += connection_pair_durations_con_iter[0]
                connection_pair_durations_con_iter.pop(0)

            if to_merge_discon:
                connection_pair_durations_discon[-1] += connection_pair_durations_discon_iter[0]
                connection_pair_durations_discon_iter.pop(0)

            connection_pair_durations_con += connection_pair_durations_con_iter
            connection_pair_durations_discon += connection_pair_durations_discon_iter

        durations_matrix_con[idx_link] = connection_pair_durations_con
        durations_matrix_discon[idx_link] = connection_pair_durations_discon

    durations_con = [item for sublist in durations_matrix_con.tolist() for item in sublist]
    durations_discon = [item for sublist in durations_matrix_discon.tolist() for item in sublist]

    connection_durations = LinkDurations(durations_con=durations_con,
                                         durations_discon=durations_discon,
                                         durations_matrix_con=durations_matrix_con,
                                         durations_matrix_discon=durations_matrix_discon)

    return connection_durations


def calc_connection_stats(durations, count_nodes):
    """Determines the average number of connected periods and the average duration of a connected period from
    durations"""

    count_pairs = count_nodes * (count_nodes - 1) // 2
    mean_duration = np.mean(durations)
    mean_connected_periods = len(durations) / count_pairs

    return mean_duration, mean_connected_periods
