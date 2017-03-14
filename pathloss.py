""" Provides pathloss functions as defined in :
Abbas, Taimoor, et al.
"A measurement based shadow fading model for vehicle-to-vehicle network simulations."
International Journal of Antennas and Propagation 2015 (2015).
"""

import numpy as np


class Pathloss:
    """ Class providing the pathloss functions"""

    nlos_config = {
        'width_rx_street': 7.5,
        'dist_tx_wall': 3.75,
        'is_sub_urban': False,
        'pathloss_exp': 2.69,
        'dist_break': 161}  # TODO: check paper for value

    los_config = {
        'dist_ref': 10,
        'dist_break': 161,
        'pathloss_exp_1': -1.81,
        'pathloss_exp_2': -2.85,
        'pathloss_ref': -63.9,
        'standard_dev': 4.15}

    olos_config = {
        'dist_ref': 10,
        'dist_break': 161,
        'pathloss_exp_1': -1.93,
        'pathloss_exp_2': -2.74,
        'pathloss_ref': -72.3,
        'standard_dev': 6.67}

    def __init__(self, wavelength=0.050812281):
        self.wavelength = wavelength

    def pathloss_nlos(self, dist_rx, dist_tx):
        """Calculates the pathloss for the non line of sight case in equation (6)"""

        # TODO: missing in equation?
        sf_loss = np.random.normal(0, self.nlos_config['standard_dev'], 1)

        if dist_rx < self.nlos_config['dist_break']:
            pathloss = 3.75 + self.nlos_config['is_sub_urban'] * 2.94 + 10 * np.log10((dist_tx ^ 0.957 * 4 * np.pi * dist_rx(
                self.nlos_config['dist_tx_wall'] * self.nlos_config['width_rx_street']) ^ 0.81 / self.wavelength) ^ self.nlos_config['pathloss_exp']) + sf_loss
        else:
            pathloss = 3.75 + self.nlos_config['is_sub_urban'] * 2.94 + 10 * np.log10((dist_tx ^ 0.957 * 4 * np.pi * dist_rx ^ 2 * (self.nlos_config['dist_tx_wall'] * self.nlos_config[
                                                                                      'width_rx_street']) ^ 0.81 / (self.wavelength * self.nlos_config['dist_break'])) ^ self.nlos_config['pathloss_exp']) + sf_loss

        return pathloss

    def pathloss_los(self, dist):
        """Calculates the pathloss for the line of sight case defined in equation (5)"""

        sf_loss = np.random.normal(0, self.los_config['standard_dev'], 1)

        if dist < self.los_config['dist_ref']:
            raise ValueError('distance smaller than reference distance')
        elif dist < self.los_config['dist_break']:
            pathloss = self.los_config['pathloss_ref'] + 10 * self.los_config['pathloss_exp_1'] * \
                np.log10(dist / self.los_config['dist_ref']) + sf_loss
        else:
            pathloss = self.los_config['pathloss_ref'] + 10 * self.los_config['pathloss_exp_1'] \
                * np.log10(self.los_config['dist_break'] / self.los_config['dist_ref']) \
                + 10 * self.los_config['pathloss_exp_2'] \
                * np.log10(dist / self.los_config['dist_break']) + sf_loss

        return pathloss

    def pathloss_olos(self, dist):
        """Calculates the pathloss for the obstructed line of sight case defined in equation (5)"""

        sf_loss = np.random.normal(0, self.olos_config['standard_dev'], 1)

        if dist < self.olos_config['dist_ref']:
            raise ValueError('distance smaller than reference distance')
        elif dist < self.olos_config['dist_break']:
            pathloss = self.olos_config['pathloss_ref'] + 10 * self.olos_config['pathloss_exp_1'] * \
                np.log10(dist / self.olos_config['dist_ref']) + sf_loss
        else:
            pathloss = self.olos_config['pathloss_ref'] + 10 * self.olos_config['pathloss_exp_1'] * \
                np.log10(self.olos_config['dist_break'] / self.olos_config['dist_ref']) \
                + 10 * self.olos_config['pathloss_exp_2'] * \
                np.log10(dist / self.olos_config['dist_break']) + sf_loss

        return pathloss
