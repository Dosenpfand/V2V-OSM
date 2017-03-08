"""Calculates the number of connections in a Manhatten grid"""
import numpy as np
from matplotlib import pyplot as plt


def count_cons_orth_streets(reps, lam_v, lam_s, road_len, com_ranges):
    """Calculates the number of connections on the orthogonal streets"""
    counts_range = np.zeros((np.size(com_ranges), reps))

    for rep in np.arange(reps):
        print(rep)
        # TODO: check!
        coords_own = np.array([road_len / 2, road_len / 2])
        # Truncated poisson variable realization
        count_streets = 0
        while count_streets == 0:
            count_streets = np.random.poisson(lam_s * road_len, 1)
        coords_streets = np.sort(np.random.uniform(0, road_len, count_streets))
        counts_veh = np.zeros(count_streets, dtype=int)

        for i_street in np.arange(0, count_streets - 1):
            while counts_veh[i_street] == 0:
                counts_veh[i_street] = np.random.poisson(lam_s * road_len, 1)

        for i_count, count_veh in np.ndenumerate(counts_veh):
            coords_veh_street = np.zeros((count_veh, 2))
            coords_veh_street[:, 0] = coords_streets[i_count]
            coords_veh_street[:, 1] = np.random.uniform(0, road_len, count_veh)
            distances = np.linalg.norm(coords_own - coords_veh_street, axis=1)

            for i_com_range, com_range in np.ndenumerate(com_ranges):
                count_range = np.count_nonzero(distances < com_range)
                counts_range[i_com_range, rep] += count_range

    mean_cons = np.mean(counts_range, 1)

    # Analytical result
    mean_cons_ana = com_ranges**2*lam_v*lam_s*np.pi/4

    # Plot error
    # TODO: results do not match yet!
    plt.figure()
    plt.plot(com_ranges, mean_cons_ana - mean_cons)
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Error')
    plt.grid(True)

    # Plot last realization of last street
    # TODO: !
    plt.figure()
    plt.scatter(coords_veh_street[:,0], coords_veh_street[:,1])
    plt.scatter(coords_own[0], coords_own[1])

    # Plot connected vehicles vs. sensing range
    plt.figure()
    plt.plot(com_ranges, mean_cons, label='Numerical')
    plt.plot(com_ranges, mean_cons_ana, label='Analytical')
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Mean connected vehicles')
    plt.legend(loc='best')
    plt.grid(True)

    # Show all plots
    plt.show()

def count_cons_own_street(reps, lam_v, road_len, com_ranges):
    """Calculates the number of connections on the street the vehicle is itself on"""
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
            counts_range[i_com_range, rep] = np.count_nonzero(
                distances < com_range) - 1

    mean_cons = np.mean(counts_range, 1)

    # Analytical result
    mean_cons_ana = 2 * lam_v * com_ranges

    # Plot error
    plt.figure()
    plt.plot(com_ranges, mean_cons_ana - mean_cons)
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Error')
    plt.grid(True)

    # Plot last realization
    plt.figure()
    plt.scatter(coords_veh, np.zeros_like(coords_veh))
    plt.scatter(coords_veh_center, 0)

    # Plot connected vehicles vs. sensing range
    plt.figure()
    plt.plot(com_ranges, mean_cons, label='Numerical')
    plt.plot(com_ranges, mean_cons_ana, label='Analytical')
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Mean connected vehicles')
    plt.legend(loc='best')
    plt.grid(True)

    # Show all plots
    plt.show()

if __name__ == '__main__':
    REPS = 1000  # Monte Carlo runs
    LAM_V = 1e-2  # (Relative) vehicle rate
    LAM_S = 1e-3  # (Relative) street rate
    ROAD_LEN = 5000  # length of roads
    COM_RANGES = np.arange(1, 1000, 10)  # range of communication

    count_cons_own_street(REPS, LAM_V, ROAD_LEN, COM_RANGES)
    count_cons_orth_streets(REPS, LAM_V, LAM_S, ROAD_LEN, COM_RANGES)
