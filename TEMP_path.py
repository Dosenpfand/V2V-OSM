import vtovosm as vtv
import networkx as nx
import numpy as np
import time

res_p = 'results/default.100.pickle.xz'

res = vtv.utils.load(res_p)

matrices_cons = res['results']['matrices_cons']
graphs = [nx.from_numpy_matrix(matrix_cons) for matrix_cons in matrices_cons]

# Method 1: nx.all_pairs_shortest_path_length

res_met_1 = []
time_start = time.process_time()

for graph in graphs:
    count_nodes = graph.number_of_nodes()
    size_cond = count_nodes * (count_nodes - 1) // 2
    has_path_matrix = np.zeros(size_cond, dtype=bool)

    path_lengths = nx.all_pairs_shortest_path_length(graph)

    for node_u, nodes_v in path_lengths.items():
        for node_v in nodes_v.keys():
            if node_u == node_v:
                continue
            idx_cond = vtv.utils.square_to_condensed(node_v, node_u, count_nodes)
            has_path_matrix[idx_cond] = True

    res_met_1.append(has_path_matrix)

time_run = time.process_time() - time_start
print(time_run)

# Method 2: Own

res_met_2 = []
time_start = time.process_time()

for graph in graphs:
    count_nodes = graph.number_of_nodes()
    size_cond = count_nodes * (count_nodes - 1) // 2
    has_path_matrix = np.zeros(size_cond, dtype=bool)

    for idx_u, node_u in enumerate(graph.nodes()):
        for idx_v, node_v in enumerate(graph.nodes()[idx_u + 1:]):
            idx_cond = vtv.utils.square_to_condensed(node_u, node_v, count_nodes)
            is_connected = nx.has_path(graph, node_u, node_v)
            if is_connected:
                has_path_matrix[idx_cond] = True

    res_met_2.append(has_path_matrix)

time_run = time.process_time() - time_start
print(time_run)

for mat_met_1, mat_met_2 in zip(res_met_1, res_met_2):
    if not np.array_equal(mat_met_1, mat_met_2):
        raise RuntimeError('Arrays not equal')
