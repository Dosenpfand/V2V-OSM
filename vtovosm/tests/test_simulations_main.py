"""Unit tests for the module simulations.main which execute slow"""

import json
import os
import unittest

import vtovosm.simulations.main as main_sim


class TestSimulationsMain(unittest.TestCase):
    """Provides unit tests for the simulations.main module"""

    module_path = os.path.dirname(__file__)
    conf_file_path = os.path.join(module_path, 'network_config', 'tests.json')

    def test_main(self):
        """Tests the function main"""

        with open(self.conf_file_path, 'r') as file:
            config_all = json.load(file)

        for scenario in config_all:
            if scenario == 'global':
                continue

            main_sim.main(conf_path=self.conf_file_path, scenario=scenario)


if __name__ == '__main__':
    unittest.main()
