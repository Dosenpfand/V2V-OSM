""" Provides pathloss functions as defined in :
Abbas, Taimoor, et al.
"A measurement based shadow fading model for vehicle-to-vehicle network simulations."
International Journal of Antennas and Propagation 2015 (2015).
"""

import logging
import numpy as np


class Pathloss:
    """ Class providing the pathloss functions for NLOS, OLOS and LOS propagation"""

    def __init__(self, nlos_config=None, los_config=None, olos_config=None):
        if nlos_config is None:
            # NOTE: dis_break assumes a vehicle height of 1.5 m, and a
            # frequency of 5.9 GHz
            self.nlos_config = {
                'wavelength': 0.050812281,
                'width_rx_street': 10,
                'dist_tx_wall': 5,
                'is_sub_urban': False,
                'pathloss_exp': 2.69,
                'dist_break': 44.25,  # TODO: check paper for value
                'standard_dev': 4.1}
        else:
            self.nlos_config = nlos_config

        if los_config is None:
            self.los_config = {
                'dist_ref': 10,
                'dist_break': 161,
                'pathloss_exp_1': -1.81,
                'pathloss_exp_2': -2.85,
                'pathloss_ref': -63.9,
                'standard_dev': 4.15}
        else:
            self.los_config = los_config

        if olos_config is None:
            self.olos_config = {
                'dist_ref': 10,
                'dist_break': 161,
                'pathloss_exp_1': -1.93,
                'pathloss_exp_2': -2.74,
                'pathloss_ref': -72.3,
                'standard_dev': 6.67}
        else:
            self.olos_config = olos_config

    # TODO: 3 setter functions for configs

    def pathloss_nlos(self, dist_rx, dist_tx):
        """Calculates the pathloss for the non line of sight case in equation (6)"""

        # TODO: missing in equation?
        sf_loss = np.random.normal(
            0, self.nlos_config['standard_dev'], np.size(dist_rx))

        # TODO: check equation again!
        slope_selector = dist_rx < self.nlos_config['dist_break']
        pathloss_slope_1 = slope_selector \
            * (3.75 + self.nlos_config['is_sub_urban'] * 2.94 + 10
               * np.log10((dist_tx ** 0.957 * 4 * np.pi * dist_rx * (self.nlos_config['dist_tx_wall']
                                                                     * self.nlos_config['width_rx_street']) ** 0.81 / self.nlos_config['wavelength'])
                          ** self.nlos_config['pathloss_exp']))
        pathloss_slope_2 = np.invert(slope_selector) \
            * (3.75 + self.nlos_config['is_sub_urban'] * 2.94 + 10
               * np.log10((dist_tx ** 0.957 * 4 * np.pi * dist_rx ** 2
                           * (self.nlos_config['dist_tx_wall'] * self.nlos_config['width_rx_street']) ** 0.81
                           / (self.nlos_config['wavelength'] * self.nlos_config['dist_break']))
                          ** self.nlos_config['pathloss_exp']))

        pathloss = pathloss_slope_1 + pathloss_slope_2 + sf_loss
        return pathloss

    def pathloss_los(self, dist):
        """Calculates the pathloss for the line of sight case defined in equation (5)"""

        sf_loss = np.random.normal(
            0, self.los_config['standard_dev'], np.size(dist))

        if (np.isscalar(dist) and dist < self.los_config['dist_ref']) or \
                (not np.isscalar(dist) and any(dist < self.los_config['dist_ref'])):
            logging.warning('Distance smaller than reference distance')

        # TODO: check equation again!
        slope_selector = dist < self.los_config['dist_break']
        pathloss_slope_1 = slope_selector \
            * (self.los_config['pathloss_ref'] + 10 * self.los_config['pathloss_exp_1']
               * np.log10(dist / self.los_config['dist_ref']))
        pathloss_slope_2 = np.invert(slope_selector) \
            * (self.los_config['pathloss_ref'] + 10 * self.los_config['pathloss_exp_1']
               * np.log10(self.los_config['dist_break'] / self.los_config['dist_ref'])
               + 10 * self.los_config['pathloss_exp_2']
               * np.log10(dist / self.los_config['dist_break']))

        # NOTE: Invert sign to keep consistency with NLOS
        pathloss = - (pathloss_slope_1 + pathloss_slope_2 + sf_loss)
        return pathloss

    def pathloss_olos(self, dist):
        """Calculates the pathloss for the obstructed line of sight case defined in equation (5)"""

        sf_loss = np.random.normal(
            0, self.olos_config['standard_dev'], np.size(dist))

        if (np.isscalar(dist) and dist < self.olos_config['dist_ref']) or \
                (not np.isscalar(dist) and any(dist < self.olos_config['dist_ref'])):
            logging.warning('Distance smaller than reference distance')

        # TODO: check equation again
        slope_selector = dist < self.olos_config['dist_break']
        pathloss_slope_1 = slope_selector \
            * (self.olos_config['pathloss_ref'] + 10 * self.olos_config['pathloss_exp_1']
               * np.log10(dist / self.olos_config['dist_ref']))
        pathloss_slope_2 = np.invert(slope_selector) \
            * (self.olos_config['pathloss_ref'] + 10 * self.olos_config['pathloss_exp_1']
               * np.log10(self.olos_config['dist_break'] / self.olos_config['dist_ref'])
               + 10 * self.olos_config['pathloss_exp_2'] *
               np.log10(dist / self.olos_config['dist_break']))

        # NOTE: Invert sign to keep consistency with NLOS
        pathloss = - (pathloss_slope_1 + pathloss_slope_2 + sf_loss)
        return pathloss
