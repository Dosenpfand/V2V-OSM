""" Provides pathloss functions as defined in :
Abbas, Taimoor, et al.
"A measurement based shadow fading model for vehicle-to-vehicle network simulations."
International Journal of Antennas and Propagation 2015 (2015).
"""

import numpy as np

def pathloss_nlos(dist_rx, dist_tx, width_rx_street, dist_tx_wall, is_sub_urban=False, wavelength=0.050812281):
    """Calculates the pathloss for the non line of sight case in equation (6)"""
    pathloss_exp = 2.69
    standard_dev = 4.1
    dist_break = 161 # TODO: check paper for value

    sf_loss = np.random.normal(0, standard_dev, 1) # TODO: missing in equation?

    wavelength = 0.05084745762711865 # corresponds to 5.9 GHz
    if dist_rx < dist_break:
        pathloss = 3.75 + is_sub_urban*2.94 + 10*np.log10((dist_tx^0.957 * 4 * np.pi * dist_rx / \
         (dist_tx_wall*width_rx_street)^0.81 / wavelength)^pathloss_exp) + sf_loss
    else:
        pathloss = 3.75 + is_sub_urban*2.94 + 10*np.log10((dist_tx^0.957 * 4 * np.pi * dist_rx^2 / \
        (dist_tx_wall*width_rx_street)^0.81 / (wavelength*dist_break))^pathloss_exp) + sf_loss

    return pathloss

def pathloss_los(dist):
    """Calculates the pathloss for the line of sight case defined in equation (5)"""
    dist_ref = 10
    dist_break = 161
    pathloss_exp_1 = -1.81
    pathloss_exp_2 = -2.85
    pathloss_ref = -63.9
    standard_dev = 4.15

    sf_loss = np.random.normal(0, standard_dev, 1)

    if dist < dist_ref:
        raise ValueError('distance smaller than reference distance')
    elif dist < dist_break:
        pathloss = pathloss_ref + 10 * pathloss_exp_1 * \
            np.log10(dist / dist_ref) + sf_loss
    else:
        pathloss = pathloss_ref + 10 * pathloss_exp_1 * np.log10(dist_break / dist_ref) \
            + 10 * pathloss_exp_2 * np.log10(dist / dist_break) + sf_loss

    return pathloss

def pathloss_olos(dist):
    """Calculates the pathloss for the obstructed line of sight case defined in equation (5)"""
    dist_ref = 10
    dist_break = 161
    pathloss_exp_1 = -1.93
    pathloss_exp_2 = -2.74
    pathloss_ref = -72.3
    standard_dev = 6.67

    sf_loss = np.random.normal(0, standard_dev, 1)

    if dist < dist_ref:
        raise ValueError('distance smaller than reference distance')
    elif dist < dist_break:
        pathloss = pathloss_ref + 10 * pathloss_exp_1 * \
            np.log10(dist / dist_ref) + sf_loss
    else:
        pathloss = pathloss_ref + 10 * pathloss_exp_1 * np.log10(dist_break / dist_ref) \
            + 10 * pathloss_exp_2 * np.log10(dist / dist_break) + sf_loss

    return pathloss
