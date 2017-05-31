"""Unit tests for the module simulations.tolerance_inspection which execute slow"""

import os
import unittest

import vtovosm.simulations.main as main_sim
import vtovosm.simulations.tolerance_inspection as tol_insp


class TestSimulationsToleranceInspection(unittest.TestCase):
    """Provides unit tests for the simulations.tolerance_inspection module"""

    module_path = os.path.dirname(__file__)
    conf_file_path = os.path.join(module_path, 'network_config', 'tolerance_inspection.json')

    def analyze_tolerance(self):
        """Tests the function analyze_tolerance"""

        main_sim.main_multi_scenario(conf_path=self.conf_file_path)
        results = tol_insp.analyze_tolerance(self.conf_file_path)

        # TODO: check results!


if __name__ == '__main__':
    unittest.main()
