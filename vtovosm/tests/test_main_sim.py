"""Unit tests for the module sim_main which execute slow"""

import json
import unittest
import os

import vtovosm.simulations.main as main_sim
import vtovosm.network_parser as nw_p


class TestSimMain(unittest.TestCase):
    """Provides unit tests for the sim_main module"""

    conf_file_path = os.path.join(nw_p.DEFAULT_CONFIG_DIR, 'tests.json')

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
