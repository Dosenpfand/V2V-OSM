"""Unit tests for all modules of the package"""

import unittest
import propagation as prop
import shapely.geometry as geom
import geometry as geom_o
import numpy as np
import networkx as nx
import geopandas as gpd


class DemoNetwork:
    """Provided functions to generate exemplary street networks and buildings"""

    def build_graph_streets(self):
        """Returns a very simple street network"""

        nodes_coords = [[0, 0],
                        [80, 0],
                        [160, 0],
                        [80, 0],
                        [80, 80],
                        [160, 80],
                        [80, 140],
                        [160, 140],
                        [80, 200],
                        [160, 200]]
        edges = [(0, 1), (1, 2), (1, 4), (2, 1), (3, 4),
                 (4, 5), (5, 4), (6, 4), (7, 6), (6, 8),
                 (8, 6), (8, 9), (9, 8)]

        graph_streets = nx.MultiDiGraph()

        for idx, node_coords in enumerate(nodes_coords):
            attrs = {'x': node_coords[0],
                     'y': node_coords[1]}
            graph_streets.add_node(idx, attr_dict=attrs)

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

    def test_upper_veh_cons_are_olos_all(self):
        """Tests the function upper_veh_cons_are_olos_all"""

        points = np.array([geom.Point(0, 0),
                           geom.Point(1, 0),
                           geom.Point(0, 1),
                           geom.Point(1, 1),
                           geom.Point(0, 2),
                           geom.Point(1, 2)], dtype=object)
        margin = 0.5
        expected_result = np.array(
            [0, 0, 0, 1, 1,
             0, 0, 1, 1,
             0, 0, 0,
             0, 0,
             0], dtype=bool)
        result = prop.veh_cons_are_olos_all(points, margin=margin)

        self.assertTrue(np.array_equal(result, expected_result))

    def test_veh_cons_are_nlos_all(self):
        """Tests the function of veh_cons_are_nlos_all"""

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)

        vehs_coords = [[40, 0], [120, 0], [80, 40], [40, 80],
                       [120, 80], [80, 120], [120, 140],
                       [80, 170], [120, 200]]
        is_nlos_expected = np.array([
            0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 1,
            1, 1, 1, 1,
            0, 0, 1,
            0, 1,
            0], dtype=bool)

        vehs_points = np.zeros(len(vehs_coords), dtype=object)
        for idx, veh_coords in enumerate(vehs_coords):
            veh_point = geom.Point(veh_coords)
            vehs_points[idx] = veh_point

        is_nlos_generated = prop.veh_cons_are_nlos_all(
            vehs_points, gdf_buildings)

        result_correct = np.array_equal(is_nlos_generated, is_nlos_expected)

        self.assertTrue(result_correct)

    def test_veh_cons_are_nlos(self):
        """Tests the function veh_cons_are_nlos"""

        network = DemoNetwork()
        graph_streets = network.build_graph_streets()
        gdf_buildings = network.build_gdf_buildings()
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave,
                              gdf_buildings,
                              max_distance=70)

        vehs_coords = [[40, 0], [120, 0], [80, 40], [40, 80],
                       [120, 80], [80, 120], [120, 140],
                       [80, 170], [120, 200]]
        is_nlos_expected = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1], dtype=bool)
        point_own = geom.Point(80, 80)

        vehs_points = np.zeros(len(vehs_coords), dtype=object)
        for idx, veh_coords in enumerate(vehs_coords):
            veh_point = geom.Point(veh_coords)
            vehs_points[idx] = veh_point

        is_nlos_generated = prop.veh_cons_are_nlos(
            point_own, vehs_points, gdf_buildings)

        result_correct = np.array_equal(is_nlos_generated, is_nlos_expected)

        self.assertTrue(result_correct)


if __name__ == '__main__':
    unittest.main()
