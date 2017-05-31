"""Unit tests for all modules that interact with SUMO and therefore execute slower and need SUMO installed"""

import unittest

import numpy as np

import vtovosm.sumo as sumo

# Setup osmnx non verbose

class TestSumo(unittest.TestCase):
    """Provides unit tests for the sumo module"""

    def test_simple_wrapper(self):
        """Tests the function simple_wrapper"""

        place = 'Salmannsdorf - Vienna - Austria'

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
            directory='sumo_data',
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
            directory='sumo_data',
            skip_if_exists=True,
            veh_class='passenger'
        )

        self.assertIsInstance(traces, np.ndarray)


if __name__ == '__main__':
    unittest.main()
