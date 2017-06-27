"""Unit tests for the module simulations.main which execute slow"""

import json
import os
import unittest

import vtovosm.simulations.main as main_sim


class TestSimulationsMain(unittest.TestCase):
    """Provides unit tests for the simulations.main module"""

    slow = True
    network = True

    module_path = os.path.dirname(__file__)
    conf_file_path = os.path.join(module_path, 'network_config', 'tests.json')

    def test_main(self):
        """Tests the function main"""

        main_sim.main_multi_scenario(conf_path=self.conf_file_path)


if __name__ == '__main__':
    unittest.main()
