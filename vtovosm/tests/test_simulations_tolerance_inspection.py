"""Unit tests for the module simulations.tolerance_inspection which execute slow"""

import os
import unittest

import vtovosm.simulations.main as main_sim
import vtovosm.simulations.tolerance_inspection as tol_insp


class TestSimulationsToleranceInspection(unittest.TestCase):
    """Provides unit tests for the simulations.tolerance_inspection module"""

    max_diff_ratio = 1e-4

    module_path = os.path.dirname(__file__)
    conf_file_path = os.path.join(module_path, 'network_config', 'tolerance_inspection.json')

    def test_analyze_tolerance(self):
        """Tests the function analyze_tolerance"""

        main_sim.main_multi_scenario(conf_path=self.conf_file_path)
        all_results = tol_insp.analyze_tolerance(self.conf_file_path)

        for results in all_results.values():
            for result in results:
                self.assertTrue(result['ratio_con_diff'] < self.max_diff_ratio)


if __name__ == '__main__':
    unittest.main()
