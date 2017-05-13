"""Unit tests for all modules of the package"""

import unittest
import propagation as prop
import shapely.geometry as geom
import geometry as geom_o
import numpy as np
import numpy.matlib
import networkx as nx
import geopandas as gpd
import vehicles


class DemoNetwork:
    """Provided functions to generate exemplary street networks and buildings"""

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
                            [[100, 20], [200, 20], [200, 60], [100, 60]],
                            [[100, 100], [140, 100], [140, 120], [100, 120]],
                            [[100, 160], [200, 160], [200, 180], [100, 180]])

        buildings = {}
        for idx, building_coords in enumerate(buildings_coords):
            building_polygon = geom.Polygon(building_coords)
            building = {'id': idx, 'geometry': building_polygon}
            buildings[idx] = building

        gdf_buildings = gpd.GeoDataFrame(buildings).T

        return gdf_buildings

    def build_vehs(self, graph_streets=None, only_coords=False):
        """ Returns vehicle coordinates if only_coords is True,
        returns vehicle points if graph_streets is None and only_coords is False
        returns vehicle object if graph_streets is not None and only_coords is False"""

        vehs_coords = [[40, 0], [120, 0], [80, 40], [40, 80],
                       [120, 80], [80, 110], [120, 140], [80, 170], [120, 200]]

        if only_coords:
            return vehs_coords

        vehs_points = np.zeros(len(vehs_coords), dtype=object)
        for idx, veh_coords in enumerate(vehs_coords):
            veh_point = geom.Point(veh_coords)
            vehs_points[idx] = veh_point

        if graph_streets is None:
            return vehs_points

        vehs = vehicles.generate_vehs(
            graph_streets, points_vehs_in=vehs_points)

        return vehs


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


class TestPropagation(unittest.TestCase):
    """Provides unit tests for the propagation module"""

    def test_gen_prop_cond_matrix(self):
        """Tests the function gen_prop_cond_matrix"""

        nlos_o = prop.Cond.NLOS_ort
        nlos_p = prop.Cond.NLOS_par
        olos = prop.Cond.OLOS
        los = prop.Cond.LOS

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

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)

        vehs = network.build_vehs(graph_streets=graph_streets)

        prop_cond_matrix_generated = prop.gen_prop_cond_matrix(
            vehs.get_points(),
            gdf_buildings,
            graph_streets_wave=graph_streets_wave,
            graphs_vehs=vehs.get_graph(),
            fully_determine=True,
            max_dist=None,
            car_radius=2,
            max_angle=np.pi / 2)

        result_correct = np.array_equal(
            prop_cond_matrix_generated,
            prop_cond_matrix_expected)
        self.assertTrue(result_correct)

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
