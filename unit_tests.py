"""Unit tests for all modules of the package"""

import unittest
import propagation as prop
import shapely.geometry as geom
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


if __name__ == '__main__':
    unittest.main()
