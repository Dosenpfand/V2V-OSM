"""Unit tests for all modules that execute fast"""

import os
import pickle
import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.matlib
import scipy.spatial.distance as sp_dist
import shapely.geometry as geom

import vtovosm.connection_analysis as con_ana
import vtovosm.geometry as geom_o
import vtovosm.pathloss as pathloss
import vtovosm.propagation as prop
import vtovosm.utils as utils
import vtovosm.vehicles as vehicles


class DemoNetwork:
    """Provides functions to generate exemplary street networks and buildings"""

    def build_graph_streets(self):
        """Returns a very simple street network"""

        nodes_coords = [[0, 0],
                        [80, 0],
                        [160, 0],
                        [0, 80],
                        [80, 80],
                        [160, 80],
                        [80, 140],
                        [160, 140],
                        [80, 200],
                        [160, 200]]
        edges = [[0, 1], [1, 2], [1, 4], [2, 1], [3, 4],
                 [4, 5], [5, 4], [6, 4], [7, 6], [6, 8],
                 [8, 6], [8, 9], [9, 8]]

        graph_streets = nx.MultiDiGraph()

        for idx, node_coords in enumerate(nodes_coords):
            attrs = {'x': node_coords[0],
                     'y': node_coords[1]}
            graph_streets.add_node(idx, attr_dict=attrs)

        for edge in edges:
            coords_1 = [graph_streets.node[edge[0]]['x'],
                        graph_streets.node[edge[0]]['y']]
            coords_2 = [graph_streets.node[edge[1]]['x'],
                        graph_streets.node[edge[1]]['y']]
            edge_geometry = geom.LineString([coords_1, coords_2])
            edge_length = edge_geometry.length
            edge.append({'geometry': edge_geometry,
                         'length': edge_length})

        graph_streets.add_edges_from(edges)

        return graph_streets

    def build_gdf_buildings(self):
        """Returns a very simple geodataframe with polygons representing builings"""

        buildings_coords = ([[20, 20], [60, 20], [60, 60], [20, 60]],
                            [[100, 20], [240, 20], [240, 60], [220, 60], [
                                220, 40], [200, 40], [200, 60], [100, 60]],
                            [[100, 100], [140, 100], [140, 120], [100, 120]],
                            [[100, 160], [200, 160], [200, 180], [100, 180]])

        buildings = {}
        for idx, building_coords in enumerate(buildings_coords):
            building_polygon = geom.Polygon(building_coords)
            building = {'id': idx, 'geometry': building_polygon}
            buildings[idx] = building

        gdf_buildings = gpd.GeoDataFrame(buildings).T

        return gdf_buildings

    def build_gdf_boundary(self):
        """Returns the boundary of the network as a GeoDataFrame"""

        polygon = geom.Polygon([(0, 0), (160, 0), (160, 200), (0, 200)])
        gdf_boundary = gpd.GeoDataFrame({0: {'id': 0, 'geometry': polygon}}).T

        return gdf_boundary

    def build_vehs(self, graph_streets=None, only_coords=False):
        """ Returns vehicle coordinates if only_coords is True,
        returns vehicle points if graph_streets is None and only_coords is False
        returns vehicle object if graph_streets is not None and only_coords is False"""

        vehs_coords = [[40, 0], [120, 0], [80, 40], [40, 80],
                       [120, 80], [80, 110], [120, 140], [80, 170], [120, 200]]

        if only_coords:
            return np.array(vehs_coords)

        vehs_points = np.zeros(len(vehs_coords), dtype=object)
        for idx, veh_coords in enumerate(vehs_coords):
            veh_point = geom.Point(veh_coords)
            vehs_points[idx] = veh_point

        if graph_streets is None:
            return vehs_points

        vehs = vehicles.generate_vehs(
            graph_streets, points_vehs_in=vehs_points)

        return vehs


class Pathloss():
    """Provides reimplementations of the pathloss functions to test against"""

    @staticmethod
    def get_pathloss_olos_los_urban(dist, is_LOS=True):
        """Determine the pathloss for OLOS/LOS propagation in an urban scenario"""

        dist_ref = 10
        dist_break = 104

        if is_LOS:
            pl_exp1 = -1.81
            pl_exp2 = -2.85
            pathloss_ref = -63.9
        else:
            pl_exp1 = -1.93
            pl_exp2 = -2.74
            pathloss_ref = -72.3

        if dist <= dist_break:
            pathloss = pathloss_ref + 10 * pl_exp1 * np.log10(dist / dist_ref)
        else:
            pathloss = pathloss_ref + 10 * pl_exp1 * np.log10(dist_break / dist_ref) + 10 * pl_exp2 * np.log10(
                dist / dist_break)

        return pathloss

    @staticmethod
    def get_pathloss_nlos_urban(dist_rx, dist_tx):
        """Determine the pathloss for NLOS propagation in an urban scenario"""

        dist_break = 44.25
        width_street = 10
        dist_wall = 5
        wavelength = 0.050812281

        if dist_rx <= dist_break:
            pathloss = 3.75 + 10 * np.log10(
                (dist_tx ** 0.957 * 4 * np.pi * dist_rx / ((dist_wall * width_street) ** 0.81 * wavelength)) ** 2.69)
        else:
            pathloss = 3.75 + 10 * np.log10(
                (dist_tx ** 0.957 * 4 * np.pi * dist_rx ** 2 / (
                    (dist_wall * width_street) ** 0.81 * wavelength * dist_break)) ** 2.69)

        return pathloss


class TestPathloss(unittest.TestCase):
    """Provides unit tests for the pathloss module"""

    def test_pathloss_los(self):
        """Tests the function pathloss_los"""

        iterations = 100
        pl = pathloss.Pathloss()
        pl.disable_shadowfading()

        # Test first slope
        dist = np.random.rand(iterations) * 100
        for idx in range(iterations):
            # NOTE: Minus to keep consistent with pathloss.py
            pathloss_expected = - Pathloss.get_pathloss_olos_los_urban(dist[idx], is_LOS=True)
            pathloss_generated = pl.pathloss_los(dist[idx])[0]
            self.assertAlmostEqual(pathloss_expected, pathloss_generated)

        # Test second slope
        dist = 100 + np.random.rand(iterations) * 5000
        for idx in range(iterations):
            # NOTE: Minus to keep consistent with pathloss.py
            pathloss_expected = - Pathloss.get_pathloss_olos_los_urban(dist[idx], is_LOS=True)
            pathloss_generated = pl.pathloss_los(dist[idx])[0]
            self.assertAlmostEqual(pathloss_expected, pathloss_generated)

        # Test arrays
        dist = np.random.rand(iterations) * 5000
        pathlosses_generated = pl.pathloss_los(dist)

        for idx in range(iterations):
            # NOTE: Minus to keep consistent with pathloss.py
            pathloss_expected = - Pathloss.get_pathloss_olos_los_urban(dist[idx], is_LOS=True)
            self.assertAlmostEqual(pathloss_expected, pathlosses_generated[idx])

    def test_pathloss_olos(self):
        """Tests the function pathloss_olos"""

        iterations = 100
        pl = pathloss.Pathloss()
        pl.disable_shadowfading()

        # Test first slope
        dist = np.random.rand(iterations) * 100
        for idx in range(iterations):
            # NOTE: Minus to keep consistent with pathloss.py
            pathloss_expected = - Pathloss.get_pathloss_olos_los_urban(dist[idx], is_LOS=False)
            pathloss_generated = pl.pathloss_olos(dist[idx])[0]
            self.assertAlmostEqual(pathloss_expected, pathloss_generated)

        # Test second slope
        dist = 100 + np.random.rand(iterations) * 5000
        for idx in range(iterations):
            # NOTE: Minus to keep consistent with pathloss.py
            pathloss_expected = - Pathloss.get_pathloss_olos_los_urban(dist[idx], is_LOS=False)
            pathloss_generated = pl.pathloss_olos(dist[idx])[0]
            self.assertAlmostEqual(pathloss_expected, pathloss_generated)

        # Test arrays
        dist = np.random.rand(iterations) * 5000
        pathlosses_generated = pl.pathloss_olos(dist)

        for idx in range(iterations):
            # NOTE: Minus to keep consistent with pathloss.py
            pathloss_expected = - Pathloss.get_pathloss_olos_los_urban(dist[idx], is_LOS=False)
            self.assertAlmostEqual(pathloss_expected, pathlosses_generated[idx])

    def test_pathloss_nlos(self):
        """Tests the function pathloss_nlos"""

        iterations = 100
        pl = pathloss.Pathloss()
        pl.disable_shadowfading()

        # Test first slope
        dist_rx = np.random.rand(iterations) * 44.25
        dist_tx = np.random.rand(iterations) * 44.25
        for idx in range(iterations):
            pathloss_expected = Pathloss.get_pathloss_nlos_urban(dist_rx[idx], dist_tx[idx])
            pathloss_generated = pl.pathloss_nlos(dist_rx[idx], dist_tx[idx])[0]
            self.assertAlmostEqual(pathloss_expected, pathloss_generated)

        # Test second slope
        dist_rx = 44.25 + np.random.rand(iterations) * 5000
        dist_tx = 44.25 + np.random.rand(iterations) * 5000
        for idx in range(iterations):
            pathloss_expected = Pathloss.get_pathloss_nlos_urban(dist_rx[idx], dist_tx[idx])
            pathloss_generated = pl.pathloss_nlos(dist_rx[idx], dist_tx[idx])[0]
            self.assertAlmostEqual(pathloss_expected, pathloss_generated)

        # Test arrays
        dist_rx = np.random.rand(iterations) * 5000
        dist_tx = np.random.rand(iterations) * 5000
        pathlosses_generated = pl.pathloss_nlos(dist_rx, dist_tx)
        for idx in range(iterations):
            pathloss_expected = Pathloss.get_pathloss_nlos_urban(dist_rx[idx], dist_tx[idx])
            self.assertAlmostEqual(pathloss_expected, pathlosses_generated[idx])


class TestVehicles(unittest.TestCase):
    """Provides unit tests for the vehicles module"""

    def test_place_vehicles_in_network(self):
        """Tests the function place_vehicles_in_network"""

        densities = [100, 1e-1, 1e-3]
        density_types = ['absolute', 'length', 'area']
        counts_vehs_expected = [100, 980 * densities[1], 32000 * densities[2]]

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_boundary = network.build_gdf_boundary()
        net = {'graph_streets': graph_streets,
               'gdf_boundary': gdf_boundary}

        for density, density_type, count_vehs_expected in \
                zip(densities, density_types, counts_vehs_expected):
            vehs = vehicles.place_vehicles_in_network(
                net, density_veh=density, density_type=density_type)
            count_correct = vehs.count == count_vehs_expected
            self.assertTrue(count_correct)


class TestGeometry(unittest.TestCase):
    """Provides unit tests for the geometry module"""

    def test_line_intersects_buildings(self):
        """Tests the function line_intersects_buildings"""

        network = DemoNetwork()
        gdf_buildings = network.build_gdf_buildings()

        lines_coords = [[[0, 0], [160, 200]],
                        [[0, 0], [0, 80]],
                        [[0, 40], [40, 0]],
                        [[0, 40 - 1e-10], [40, 0]]]
        intersect_flags = [True, False, True, False]

        for line_coords, intersect_flag in zip(lines_coords, intersect_flags):
            line = geom.LineString(line_coords)
            intersects = geom_o.line_intersects_buildings(line, gdf_buildings)
            result_correct = intersects == intersect_flag
            self.assertTrue(result_correct)

    def test_line_intersects_points(self):
        """Tests the function line_intersects_points"""

        points_coords = [[0, 0], [2, 0], [0, 2], [2, 2]]
        lines_coords = [[[-1, -1], [1, 1]],
                        [[0, 1], [2, 1]],
                        [[0, 0], [2, 0]],
                        [[0, 0.5], [2, 0.5]],
                        [[0, 0.5 + 1e10], [2, 0.5 + 1e10]]]
        intersect_flags = [True, False, True, True, False]
        margin = 0.5

        points = []
        for point_coords in points_coords:
            points.append(geom.Point(point_coords))

        for line_coords, intersect_flag in zip(lines_coords, intersect_flags):
            line = geom.LineString(line_coords)
            intersects = geom_o.line_intersects_points(
                line, points, margin=margin)

            result_correct = intersects == intersect_flag
            self.assertTrue(result_correct)

    def test_get_street_lengths(self):
        """Tests the function get_street_lengths"""

        lengths_expected = np.append([60, 60, 60], np.matlib.repeat(80, 10))

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        lengths_generated = geom_o.get_street_lengths(graph_streets)

        results_correct = np.array_equal(
            np.sort(lengths_generated), lengths_expected)
        self.assertTrue(results_correct)
        return

    def test_find_center_veh(self):
        """Tests the function find_center_veh"""

        index_expected = 5

        network = DemoNetwork()
        vehs_coords = network.build_vehs(graph_streets=None, only_coords=True)
        index_generated = geom_o.find_center_veh(vehs_coords)

        self.assertTrue(index_expected == index_generated)

    def test_split_line_at_point(self):
        """Tests the function split_line_at_point"""

        line = geom.LineString([(0, 0), (0, 1)])
        point_success = geom.Point(0, 0.5)
        line_before_expected = geom.LineString([(0, 0), (0, 0.5)])
        line_after_expected = geom.LineString([(0, 0.5), (0, 1)])
        point_fail = geom.Point(1e-5, 0.5)

        line_before_generated, line_after_generated = geom_o.split_line_at_point(
            line, point_success)

        max_diff_before = np.max(
            np.abs(np.array(line_before_expected) - np.array(line_before_generated)))
        max_diff_after = np.max(
            np.abs(np.array(line_after_expected) - np.array(line_after_generated)))
        max_diff = np.max([max_diff_after, max_diff_before])
        result_correct = max_diff < 1e-6

        self.assertTrue(result_correct)

        with self.assertRaises(ValueError):
            geom_o.split_line_at_point(line, point_fail)


class TestUtils(unittest.TestCase):
    """Provides unit tests for the utils module"""

    def test_save_load(self):
        """Tests the functions save and load"""

        file_path = 'results/TEMP_test_load_save.pickle.xz'

        save_data = np.random.rand(100)
        utils.save(save_data, file_path)

        load_data = utils.load(file_path)

        result_correct = numpy.array_equal(load_data, save_data)
        self.assertTrue(result_correct)

        os.remove(file_path)

    def test_compress_file(self):
        """Tests the function compress_file"""

        file_path_uncomp = 'results/TEMP_test_compress_file.pickle'
        file_path_comp = file_path_uncomp + '.xz'

        save_data = np.random.rand(100)

        with open(file_path_uncomp, 'wb') as file:
            pickle.dump(save_data, file)

        utils.compress_file(file_path_uncomp)
        load_data = utils.load(file_path_comp)

        result_correct = numpy.array_equal(load_data, save_data)
        self.assertTrue(result_correct)

        os.remove(file_path_comp)

    def test_square_to_condensed_condensed_to_square(self):
        """Tests the functions square_to_condensed and condensed_to_square"""

        size_n = 7

        size_cond = size_n * (size_n - 1) // 2
        condensed = np.random.rand(size_cond)
        square = sp_dist.squareform(condensed)

        for idx_cond in range(size_cond):
            idx_i, idx_j = utils.condensed_to_square(idx_cond, size_n)
            result_correct = square[idx_i, idx_j] == condensed[idx_cond]
            self.assertTrue(result_correct)

        for idx_i in range(size_n):
            for idx_j in range(size_n):
                if idx_i == idx_j:
                    with self.assertRaises(ValueError):
                        utils.square_to_condensed(idx_i, idx_j, size_n)
                else:
                    idx_cond = utils.square_to_condensed(idx_i, idx_j, size_n)
                    result_correct = square[idx_i,
                                            idx_j] == condensed[idx_cond]
                    self.assertTrue(result_correct)


class TestConnectionAnalysis(unittest.TestCase):
    """Provides unit tests for the connection_analysis module"""

    def test_calc_connection_stats(self):
        """Tests the function calc_connection_stats"""

        durations = [1, 2, 1, 2, 4, 5, 2, 2, 2, 2]
        count_nodes = 4
        mean_duration_expected = 2.3
        mean_connected_periods_expected = 10 / 6

        mean_duration_generated, mean_connected_periods_generated = \
            con_ana.calc_connection_stats(durations, count_nodes)

        self.assertAlmostEqual(mean_duration_generated, mean_duration_expected)
        self.assertAlmostEqual(mean_connected_periods_generated, mean_connected_periods_expected)

    def test_calc_connection_durations(self):
        """Tests the function calc_connection_durations"""

        con_matrices_cond = [
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1]
        ]
        durations_con_expected = [1, 2, 1, 2, 4, 5, 2, 2, 2, 2]
        durations_discon_expected = [1, 1, 1, 1, 1, 1, 1]

        con_matrices = [sp_dist.squareform(con_matrix_cond) for con_matrix_cond in con_matrices_cond]
        graphs_cons = [nx.from_numpy_matrix(con_matrix) for con_matrix in con_matrices]

        durations_generated = con_ana.calc_connection_durations(graphs_cons)

        self.assertEqual(durations_generated[0], durations_con_expected)
        self.assertEqual(durations_generated[1], durations_discon_expected)

    def test_calc_link_durations(self):
        """Tests the function calc_link_durations"""

        con_matrices_cond = [
            [1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1]
        ]
        durations_con_expected = [4, 1, 1, 5, 2, 2]
        durations_discon_expected = [1, 5, 2, 1, 5, 1]

        con_matrices = [sp_dist.squareform(con_matrix_cond) for con_matrix_cond in con_matrices_cond]
        graphs_cons = [nx.from_numpy_matrix(con_matrix) for con_matrix in con_matrices]

        durations_generated = con_ana.calc_link_durations(graphs_cons)

        self.assertEqual(durations_generated[0], durations_con_expected)
        self.assertEqual(durations_generated[1], durations_discon_expected)

    def test_calc_link_durations_multiprocess(self):
        """Tests the function test_calc_link_durations_m"""

        con_matrices_cond = [
            [1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1]
        ]
        chunk_lengths = range(1, 4)

        con_matrices = [sp_dist.squareform(con_matrix_cond) for con_matrix_cond in con_matrices_cond]
        graphs_cons = [nx.from_numpy_matrix(con_matrix) for con_matrix in con_matrices]

        durations_singlep = con_ana.calc_link_durations(graphs_cons)

        for chunk_length in chunk_lengths:
            durations_multip = con_ana.calc_link_durations_multiprocess(graphs_cons, mp_pool=None,
                                                                        chunk_length=chunk_length)

            self.assertEqual(durations_multip.durations_con, durations_singlep.durations_con)
            self.assertEqual(durations_multip.durations_discon, durations_singlep.durations_discon)
            self.assertEqual(durations_multip.durations_matrix_con.tolist(),
                             durations_singlep.durations_matrix_con.tolist())
            self.assertEqual(durations_multip.durations_matrix_discon.tolist(),
                             durations_singlep.durations_matrix_discon.tolist())

    def test_calc_path_redundancy(self):
        """Tests the function calc_center_path_redundancy"""

        max_dist = {'nlos': 100, 'olos_los': 150}
        node_redundancies_expteced = [3, 3, 4, 5, 5, 5, 6, 3]
        edge_redundancies_expteced = [3, 3, 6, 6, 6, 5, 6, 3]
        sqrt = np.sqrt
        distances_expected = [sqrt(40 ** 2 + 110 ** 2),
                              sqrt(40 ** 2 + 110 ** 2),
                              70,
                              sqrt(30 ** 2 + 40 ** 2),
                              sqrt(30 ** 2 + 40 ** 2),
                              sqrt(30 ** 2 + 40 ** 2),
                              60,
                              sqrt(90 ** 2 + 40 ** 2)]

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)
        vehs = network.build_vehs(
            graph_streets=graph_streets, only_coords=False)

        graph_cons = con_ana.gen_connection_graph(
            vehs,
            gdf_buildings,
            max_dist,
            metric='distance',
            graph_streets_wave=graph_streets_wave)

        path_redundancy = con_ana.calc_center_path_redundancy(graph_cons, vehs)

        node_redundancy_correct = np.array_equal(
            path_redundancy['count_node_disjoint_paths'],
            node_redundancies_expteced)
        self.assertTrue(node_redundancy_correct)

        edge_redundancy_correct = np.array_equal(
            path_redundancy['count_edge_disjoint_paths'],
            edge_redundancies_expteced)
        self.assertTrue(edge_redundancy_correct)

        distance_correct = np.array_equal(
            path_redundancy['distance'],
            distances_expected
        )
        self.assertTrue(distance_correct)

    def test_calc_net_connectivity(self):
        """Tests the function calc_net_connectivity"""

        edges = [(0, 1), (0, 2), (0, 3), (0, 4),
                 (5, 6), (5, 7), (5, 8)]
        net_connectivity_expected = 5 / 9

        graph = nx.Graph()
        graph.add_edges_from(edges)
        net_connectivity_generated = con_ana.calc_net_connectivity(graph)

        self.assertAlmostEqual(net_connectivity_generated, net_connectivity_expected)

    def test_gen_connection_graph(self):
        """Tests the function gen_connection_graph"""

        # Distance config
        max_dist = {'nlos': 100, 'olos_los': 150}
        edges_expected = [(0, 1), (0, 2), (0, 3),
                          (1, 2), (1, 4),
                          (2, 3), (2, 4), (2, 5), (2, 7),
                          (3, 4), (3, 5), (3, 6), (3, 7),
                          (4, 5), (4, 6), (4, 7),
                          (5, 6), (5, 7), (5, 8),
                          (6, 7), (6, 8), (6, 9),
                          (7, 8), (7, 9),
                          (8, 9)]

        edges_expected_sets = [set(edge) for edge in edges_expected]
        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)
        vehs = network.build_vehs(
            graph_streets=graph_streets, only_coords=False)

        # Distance based connection matrix
        con_graph_dist_generated = con_ana.gen_connection_graph(
            vehs,
            gdf_buildings,
            max_dist,
            metric='distance',
            graph_streets_wave=graph_streets_wave)

        self.assertIsInstance(con_graph_dist_generated, nx.Graph)

        count_nodes = con_graph_dist_generated.number_of_nodes()

        for node_u in range(count_nodes):
            for node_v in range(count_nodes):
                edge_current = (node_u, node_v)
                if set(edge_current) in edges_expected_sets:
                    result_correct = con_graph_dist_generated.has_edge(*edge_current)
                else:
                    result_correct = not con_graph_dist_generated.has_edge(*edge_current)
                self.assertTrue(result_correct)

    def test_gen_connection_matrix(self):
        """Tests the function gen_connection_matrix"""

        # Distance config
        max_dist = {'nlos': 100, 'olos_los': 150}
        con_matrix_dist_expected_cond = np.array([
            1, 1, 1, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 0, 0,
            1, 1, 1, 0, 1, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 0,
            1, 1, 1,
            1, 1,
            1
        ], dtype=bool)

        # Pathloss config
        max_pl = 120
        con_matrix_pl_expected_cond = np.array([
            1, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 0, 1, 0,
            1, 1, 1, 1, 0,
            1, 0, 0, 0,
            1, 1, 0,
            1, 0,
            1
        ], dtype=bool)

        # Exception config
        metric_not_implemented = 'foo_bar_does_not_exist'

        con_matrix_dist_expected = sp_dist.squareform(con_matrix_dist_expected_cond)
        con_matrix_pl_expected = sp_dist.squareform(con_matrix_pl_expected_cond)

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)
        vehs = network.build_vehs(
            graph_streets=graph_streets, only_coords=False)

        # Distance based connection matrix
        con_matrix_dist_generated = con_ana.gen_connection_matrix(
            vehs,
            gdf_buildings,
            max_dist,
            metric='distance',
            graph_streets_wave=graph_streets_wave)

        result_correct = np.array_equal(
            con_matrix_dist_generated,
            con_matrix_dist_expected)
        self.assertTrue(result_correct)

        # Pathloss based connection matrix
        metric_config = {'shadowfading_enabled': False}
        con_matrix_pl_generated = con_ana.gen_connection_matrix(
            vehs,
            gdf_buildings,
            max_pl,
            metric='pathloss',
            graph_streets_wave=graph_streets_wave,
            metric_config=metric_config)

        result_correct = np.array_equal(
            con_matrix_pl_generated,
            con_matrix_pl_expected)
        self.assertTrue(result_correct)

        # Exception check
        with self.assertRaises(NotImplementedError):
            con_ana.gen_connection_matrix(
                vehs,
                gdf_buildings,
                max_dist,
                metric=metric_not_implemented,
                graph_streets_wave=graph_streets_wave)


class TestPropagation(unittest.TestCase):
    """Provides unit tests for the propagation module"""

    def test_gen_prop_cond_matrix(self):
        """Tests the function gen_prop_cond_matrix"""

        nlos_o = prop.Cond.NLOS_ort
        nlos_p = prop.Cond.NLOS_par
        olos = prop.Cond.OLOS
        los = prop.Cond.LOS
        olos_los = prop.Cond.OLOS_LOS
        nlos = prop.Cond.NLOS

        prop_cond_matrix_expected = np.array([
            los, nlos_o, nlos_p, nlos_p, nlos_o, nlos_p, nlos_o, nlos_p,
            nlos_o, nlos_p, nlos_p, nlos_o, nlos_p, nlos_o, nlos_p,
            nlos_o, nlos_o, los, nlos_o, olos, nlos_o,
            los, los, olos, los, nlos_p,
            los, nlos_p, nlos_o, nlos_p,
            los, los, nlos_o,
            los, nlos_p,
            los
        ], dtype=prop.Cond)

        arr = np.array
        coords_angle_expected = np.array([
            0, arr([80, 0]), 0, 0, arr([80, 0]), 0, arr([80, 0]), 0,
            arr([80, 0]), 0, 0, arr([80, 0]), 0, arr([80, 0]), 0,
            arr([80, 80]), arr([80, 80]), 0, arr([80, 140]), 0, arr([80, 200]),
            0, 0, 0, 0, 0,
            0, 0, arr([80, 80]), 0,
            0, 0, arr([80, 200]),
            0, 0,
            0
        ], dtype=object)

        prop_cond_matrix_red_expected = prop_cond_matrix_expected.copy()
        prop_cond_matrix_red_expected[numpy.logical_or(
            prop_cond_matrix_red_expected == nlos_o,
            prop_cond_matrix_red_expected == nlos_p)] = nlos
        prop_cond_matrix_red_expected[
            numpy.logical_or(
                prop_cond_matrix_red_expected == olos,
                prop_cond_matrix_red_expected == los)] = olos_los
        coords_angle_red_expected = np.zeros_like(
            coords_angle_expected)

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)

        vehs = network.build_vehs(graph_streets=graph_streets)

        prop_cond_matrix_generated, coords_angle_generated = \
            prop.gen_prop_cond_matrix(
                vehs.get_points(),
                gdf_buildings,
                graph_streets_wave=graph_streets_wave,
                graphs_vehs=vehs.get_graph(),
                fully_determine=True,
                max_dist=None,
                car_radius=2,
                max_angle=np.pi / 2)

        prop_cond_matrix_red_generated, coords_angle_red_generated = \
            prop.gen_prop_cond_matrix(
                vehs.get_points(),
                gdf_buildings,
                graph_streets_wave=graph_streets_wave,
                graphs_vehs=vehs.get_graph(),
                fully_determine=False,
                max_dist=None,
                car_radius=2,
                max_angle=np.pi / 2)

        result_correct = np.array_equal(
            prop_cond_matrix_generated,
            prop_cond_matrix_expected)
        self.assertTrue(result_correct)

        for generated, expected in \
                zip(coords_angle_generated, coords_angle_expected):
            result_correct = generated == expected
            self.assertTrue(numpy.all(result_correct))

        result_correct = np.array_equal(
            prop_cond_matrix_red_generated,
            prop_cond_matrix_red_expected)
        self.assertTrue(result_correct)

        for generated, expected in \
                zip(coords_angle_red_generated, coords_angle_red_expected):
            result_correct = generated == expected
            self.assertTrue(numpy.all(result_correct))

    def test_veh_cons_are_olos(self):
        """Tests the function veh_cons_are_olos"""

        vehs_coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
        point_own = geom.Point(0, -1)
        margin = 0.5
        expected_result = np.array(
            [0, 0, 1, 1, 1, 1], dtype=bool)

        vehs_points = np.zeros(len(vehs_coords), dtype=object)
        for idx, veh_coords in enumerate(vehs_coords):
            vehs_points[idx] = geom.Point(veh_coords)

        result = prop.veh_cons_are_olos(point_own, vehs_points, margin=margin)

        self.assertTrue(np.array_equal(result, expected_result))

    def test_veh_cons_are_nlos_all(self):
        """Tests the function of veh_cons_are_nlos_all"""

        is_nlos_expected = np.array([
            0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 1,
            0, 1, 1, 1,
            0, 0, 1,
            0, 1,
            0], dtype=bool)
        max_distance = 70

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=max_distance)
        vehs_points = network.build_vehs()

        is_nlos_generated = prop.veh_cons_are_nlos_all(
            vehs_points, gdf_buildings)
        result_correct = np.array_equal(is_nlos_generated, is_nlos_expected)

        self.assertTrue(result_correct)

    def test_veh_cons_are_nlos(self):
        """Tests the function veh_cons_are_nlos"""

        is_nlos_expected = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1], dtype=bool)
        point_own = geom.Point(80, 80)

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)
        vehs_points = network.build_vehs()

        is_nlos_generated = prop.veh_cons_are_nlos(
            point_own, vehs_points, gdf_buildings)
        result_correct = np.array_equal(is_nlos_generated, is_nlos_expected)

        self.assertTrue(result_correct)

    def test_check_if_cons_orthogonal(self):
        """Tests the function check_if_cons_orthogonal"""

        idx_own = 0
        is_orthogonal_expected = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
        coords_max_angle_expected = np.matlib.repmat([80, 0], 8, 1)

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)

        vehs = network.build_vehs(
            graph_streets=graph_streets, only_coords=False)
        idxs_other = np.setdiff1d(np.arange(vehs.count), idx_own)
        vehs.add_key('center', idx_own)
        vehs.add_key('other', idxs_other)

        is_orthogonal_generated, coords_max_angle_generated = \
            prop.check_if_cons_are_orthogonal(
                graph_streets_wave,
                vehs.get_graph('center'),
                vehs.get_graph('other'),
                max_angle=np.pi / 2)

        result_correct = np.array_equal(
            is_orthogonal_generated,
            is_orthogonal_expected)

        self.assertTrue(result_correct)

        result_correct = np.array_equal(
            coords_max_angle_generated,
            coords_max_angle_expected)

        self.assertTrue(result_correct)

    def test_add_edges_if_los(self):
        """Tests the function add_edges_if_los"""

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)

        self.assertTrue(graph_streets_wave.has_edge(5, 7))
        self.assertFalse(graph_streets_wave.has_edge(0, 3))
        self.assertFalse(graph_streets_wave.has_edge(7, 9))


if __name__ == '__main__':
    unittest.main()
