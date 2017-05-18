"""Unit tests for all modules that interact with OpenStreetMaps and therefore execute slower"""

import unittest
import osmnx as ox
import osmnx_addons as ox_a

class TestOsmnxAddons(unittest.TestCase):
    """Provides unit tests for the osmnx_addons module"""

    street_graph = None

    def test_0_which_result_polygon(self):
        """Tests the function which_result_polygon"""

        place = 'Salmannsdorf - Vienna - Austria'
        index = ox_a.which_result_polygon(place)

        # Check if a valid index was returned
        result_correct = isinstance(index, int)
        self.assertTrue(result_correct)

        # Try to download the street network with the returned index and save it for later
        self.street_graph = ox.graph_from_place(place, which_result=index)

    def test_1_add_check_geometry(self):
        """Tests the functions add_geometry and check_geometry"""

        # Check if we alread have a street graph
        self.assertIsNotNone(self.street_graph)

        # Add missing geometry entries
        ox_a.add_geometry(self.street_graph)

        # Check the geometry
        geometry_complete = ox_a.check_geometry(self.street_graph)
        self.assertTrue(geometry_complete)


if __name__ == '__main__':
    unittest.main()