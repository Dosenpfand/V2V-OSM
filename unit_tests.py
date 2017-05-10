"""Unit tests for all modules of the package"""

import unittest
import propagation as prop
import shapely.geometry as geom
import numpy as np


class TestPropagation(unittest.TestCase):
    """Provides unit tests for the propagation module"""

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
