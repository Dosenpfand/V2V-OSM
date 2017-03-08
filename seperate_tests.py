"""Calculates the number of connections in a Manhatten grid"""
import numpy as np
from matplotlib import pyplot as plt

def count_cons_own_street(reps, lam_v, road_len, com_ranges):
    """Calculates the number of connections on the street the vehicle is itself on"""
    mean_cons = np.zeros(np.size(com_ranges))
    counts_range = np.zeros((np.size(com_ranges), reps))

    for rep in np.arange(reps):
        print(rep)
        # Truncated poisson variable realization
        count_veh = 0
        while count_veh == 0:
            count_veh = np.random.poisson(lam_v * road_len, 1)
        coords_veh = np.sort(np.random.uniform(0, road_len, count_veh))
        median_index = np.floor_divide(np.size(coords_veh), 2)

        coords_veh_center = coords_veh[median_index]
        distances = np.abs(coords_veh - coords_veh_center)
        for i_com_range, com_range in np.ndenumerate(com_ranges):
            counts_range[i_com_range, rep] = np.count_nonzero(distances < com_range) - 1

    mean_cons = np.mean(counts_range, 1)

    # Analytical result
    lam_v_abs = 2 * lam_v * com_ranges
    mean_cons_ana = lam_v_abs / (1 - np.exp(-lam_v_abs)) - 1

    # Plot error
    plt.figure()
    plt.plot(com_ranges, mean_cons_ana - mean_cons, label='Truncated')
    plt.plot(com_ranges, lam_v_abs - mean_cons, label='Untrancated')
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.grid(True)

    # Plot last realization
    plt.figure()
    plt.scatter(coords_veh, np.zeros_like(coords_veh))
    plt.scatter(coords_veh_center, 1)

    # Plot connected vehicles vs. sensing range
    plt.figure()
    plt.plot(com_ranges, mean_cons, label='Numerical')
    plt.plot(com_ranges, mean_cons_ana, label='Analytical Truncated')
    plt.plot(com_ranges, lam_v_abs, label='Analytical Untrancated')
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Mean connected vehicles')
    plt.legend(loc='best')
    plt.grid(True)

    # Show all plots
    plt.show()

if __name__ == '__main__':
    REPS = 50000  # Monte Carlo runs
    LAM_V = 1e-2  # (Relative) vehicle rate
    ROAD_LEN = 5000  # length of roads
    COM_RANGES = np.arange(1, 1000, 5)  # range of communication

    count_cons_own_street(REPS, LAM_V, ROAD_LEN, COM_RANGES)
