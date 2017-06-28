"""Unit tests for all modules that interact with SUMO and therefore execute slower and need SUMO installed"""

import os
import unittest

import numpy as np

import vtovosm.sumo as sumo


class TestSumo(unittest.TestCase):
    """Provides unit tests for the sumo module"""

    slow = True
    network = True

    def test_download_streets_from_id(self):
        """Tests the function download_streets_from_id"""

        # OSM id for Salmannsdorf, Vienna, Austria
        osm_id = 5875884
        directory = os.path.join('sumo_data', 'tests')
        path_out = os.path.join(directory, 'test_download_streets_from_id_city.osm.xml')

        if os.path.isfile(path_out):
            os.remove(path_out)
        os.makedirs(directory, exist_ok=True)
        return_code = sumo.download_streets_from_id(osm_id, prefix='test_download_streets_from_id', directory=directory)

        self.assertIs(return_code, 0)
        self.assertTrue(os.path.isfile(path_out))

    def test_simple_wrapper(self):
        """Tests the function simple_wrapper"""

        place = 'Salmannsdorf - Vienna - Austria'
        directory = os.path.join('sumo_data', 'tests')

        # Run simulation with overwrite
        traces = sumo.simple_wrapper(
            place,
            which_result=None,
            count_veh=5,
            duration=60,
            warmup_duration=30,
            max_speed=None,
            tls_settings=None,
            fringe_factor=None,
            intermediate_points=None,
            start_veh_simult=True,
            coordinate_tls=True,
            directory=directory,
            skip_if_exists=False,
            veh_class='passenger'
        )

        self.assertIsInstance(traces, np.ndarray)

        # Run simulation without overwrite
        traces = sumo.simple_wrapper(
            place,
            which_result=None,
            count_veh=5,
            duration=60,
            warmup_duration=30,
            max_speed=None,
            tls_settings=None,
            fringe_factor=None,
            intermediate_points=None,
            start_veh_simult=True,
            coordinate_tls=True,
            directory=directory,
            skip_if_exists=True,
            veh_class='passenger'
        )

        self.assertIsInstance(traces, np.ndarray)


if __name__ == '__main__':
    unittest.main()
