"""Unit tests for all modules that interact with OpenStreetMaps and therefore execute slower"""

import unittest
import networkx as nx
import osmnx as ox
import geopandas as geop
import vtovosm.osmnx_addons as ox_a

# Setup osmnx non verbose

class TestOsmnxAddons(unittest.TestCase):
    """Provides unit tests for the osmnx_addons module"""

    place = 'Salmannsdorf - Vienna - Austria'

    def test_many(self):
        """Tests the functions which_result_polygon, add_geometry, check_geometry"""

        self.street_graph = None
        index = ox_a.which_result_polygon(self.place)

        # Check if a valid index was returned
        result_correct = isinstance(index, int)
        self.assertTrue(result_correct)

        # Try to download the street network with the returned index
        street_graph = ox.graph_from_place(self.place, which_result=index)
        self.assertIsInstance(street_graph, nx.MultiDiGraph)

        # Add missing geometry entries
        ox_a.add_geometry(street_graph)

        # Check the geometry
        geometry_complete = ox_a.check_geometry(street_graph)
        self.assertTrue(geometry_complete)

    def test_load_network(self):
        """Tests the function load_network"""

        # Load the network from the internet
        network = ox_a.load_network(self.place, which_result=None, overwrite=True)

        self.assertIsInstance(network['graph_streets'], nx.MultiDiGraph)
        self.assertIsInstance(network['graph_streets_wave'], nx.MultiGraph)
        self.assertIsInstance(network['gdf_buildings'], geop.GeoDataFrame)
        self.assertIsInstance(network['gdf_boundary'], geop.GeoDataFrame)

        # Load the network from disk
        network = ox_a.load_network(self.place, which_result=None, overwrite=False)

        self.assertIsInstance(network['graph_streets'], nx.MultiDiGraph)
        self.assertIsInstance(network['graph_streets_wave'], nx.MultiGraph)
        self.assertIsInstance(network['gdf_buildings'], geop.GeoDataFrame)
        self.assertIsInstance(network['gdf_boundary'], geop.GeoDataFrame)




if __name__ == '__main__':
    unittest.main()