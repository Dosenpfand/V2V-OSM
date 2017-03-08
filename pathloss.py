""" Provides pathloss functions as defined in :
Abbas, Taimoor, et al.
"A measurement based shadow fading model for vehicle-to-vehicle network simulations."
International Journal of Antennas and Propagation 2015 (2015).
"""

import numpy as np

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