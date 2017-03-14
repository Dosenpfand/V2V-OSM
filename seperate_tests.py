"""Calculates the number of connections in a Manhatten grid"""
import time
import numpy as np
from matplotlib import pyplot as plt
import numba


def count_connections(distances, com_ranges):
    """ Determines how many distances are smaller than the elements in com_ranges """
    counts_range = np.zeros_like(com_ranges)
    for i_com_range, com_range in np.ndenumerate(com_ranges):
        nonzero_elements = np.nonzero(distances < com_range)
        counts_range[i_com_range] = nonzero_elements[0].size
    return counts_range

fast_count_connections = numba.jit(nopython=True)(count_connections)


def array_count_connections(distances, com_ranges):
    """ Determines how many distances are smaller than the elements in com_ranges """
    com_ranges_tiled = np.tile(com_ranges, (distances.size, 1)).T
    distances_tiled = np.tile(distances, (com_ranges.size, 1))
    counts_range = np.count_nonzero(distances_tiled < com_ranges_tiled, axis=1)
    return counts_range


def count_cons_par_streets(reps, lam_v, lam_s, road_len, com_ranges):
    """Calculates the number of connections on the parallel streets"""
    counts_range = np.zeros((np.size(com_ranges), reps))

    for rep in np.arange(reps):
        print(rep)
        # Truncated poisson variable realization
        count_streets = 0
        while count_streets == 0:
            count_streets = np.random.poisson(lam_s * road_len, 1)

        coords_streets = np.sort(np.random.uniform(0, road_len, count_streets))

        # Set own position at middle of midmost street
        median_index = np.floor_divide(np.size(coords_streets), 2)
        coords_own = np.array((road_len / 2, coords_streets[median_index]))

        # Delete street we are on
        count_streets -= 1
        coords_streets = np.delete(coords_streets, median_index)

        # Truncated poisson vector realization
        counts_veh = np.zeros(count_streets, dtype=int)
        # TODO: was there a bug here! check result again!!!
        for i_street in np.arange(count_streets):
            while counts_veh[i_street] == 0:
                # TODO: There was a bug here too???
                counts_veh[i_street] = np.random.poisson(lam_v * road_len, 1)

        count_veh_all = np.sum(counts_veh)
        coords_veh_all = np.zeros((count_veh_all, 2))

        i_count_total = 0
        for i_count, count_veh in np.ndenumerate(counts_veh):
            coords_veh_street = np.zeros((count_veh, 2))
            coords_veh_street[:, 1] = coords_streets[i_count]
            coords_veh_street[:, 0] = np.random.uniform(0, road_len, count_veh)
            coords_veh_all[i_count_total:i_count_total +
                           count_veh, :] = coords_veh_street
            i_count_total += count_veh

        distances = np.linalg.norm(coords_own - coords_veh_all, ord=2, axis=1)
        counts_range[:, rep] = array_count_connections(
            distances, com_ranges)

    # Numerical result
    mean_cons = np.mean(counts_range, 1)

    # Analytical result
    # TODO: not correct yet?
    mean_cons_ana = com_ranges**2 * lam_v * lam_s * np.pi / 4
    # mean_cons_ana = (2 * com_ranges * lam_s / (1 -
    #                                            np.exp(-com_ranges * lam_s)) - 1) * np.pi / 2 * lam_v * com_ranges

    return mean_cons, mean_cons_ana, coords_veh_all, coords_own


def count_cons_orth_streets(reps, lam_v, lam_s, road_len, com_ranges):
    """Calculates the number of connections on the orthogonal streets"""
    counts_range = np.zeros((np.size(com_ranges), reps))

    for rep in np.arange(reps):
        print(rep)
        # Set own position at middle of map
        coords_own = np.array([road_len / 2, road_len / 2])
        # Truncated poisson variable realization
        count_streets = 0
        while count_streets == 0:
            count_streets = np.random.poisson(lam_s * road_len, 1)
        coords_streets = np.random.uniform(0, road_len, count_streets)
        counts_veh = np.zeros(count_streets, dtype=int)

        # Truncated poisson vector realization
        # TODO: was there a bug here! check result again!!!
        for i_street in np.arange(count_streets):
            while counts_veh[i_street] == 0:
                # TODO: There was a bug here too???
                counts_veh[i_street] = np.random.poisson(lam_v * road_len, 1)

        count_veh_all = np.sum(counts_veh)
        coords_veh_all = np.zeros((count_veh_all, 2))

        i_count_total = 0
        for i_count, count_veh in np.ndenumerate(counts_veh):
            coords_veh_street = np.zeros((count_veh, 2))
            coords_veh_street[:, 0] = coords_streets[i_count]
            coords_veh_street[:, 1] = np.random.uniform(0, road_len, count_veh)
            coords_veh_all[i_count_total:i_count_total +
                           count_veh, :] = coords_veh_street
            i_count_total += count_veh

        distances = np.linalg.norm(coords_own - coords_veh_all, ord=2, axis=1)
        counts_range[:, rep] = array_count_connections(
            distances, com_ranges)

    # Numerical result
    mean_cons = np.mean(counts_range, 1)

    # Analytical result
    # TODO: not correct yet!
    mean_cons_ana = com_ranges**2 * lam_v * lam_s * np.pi / 4

    return mean_cons, mean_cons_ana, coords_veh_all, coords_own


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

        # Set own position in the midmost car
        median_index = np.floor_divide(np.size(coords_veh), 2)
        coords_veh_center = coords_veh[median_index]
        distances = np.abs(coords_veh - coords_veh_center)
        counts_range[:, rep] = array_count_connections(
            distances, com_ranges) - 1

    mean_cons = np.mean(counts_range, 1)

    # Analytical result
    mean_cons_ana = 2 * lam_v * com_ranges

    return mean_cons, mean_cons_ana, coords_veh, coords_veh_center


def plot_results(mean_cons, mean_cons_ana, com_ranges, coords_veh_all, coords_own):
    """Plots the results returned by other functions of the package"""

    # Plot error
    plt.figure()
    plt.plot(com_ranges, mean_cons_ana - mean_cons)
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Error')
    plt.grid(True)

    # Plot last realization
    plt.figure()
    plt.scatter(coords_veh_all[:, 0], coords_veh_all[:, 1])
    plt.scatter(coords_own[0], coords_own[1])

    # Plot connected vehicles vs. sensing range
    plt.figure()
    plt.plot(com_ranges, mean_cons, label='Numerical')
    plt.plot(com_ranges, mean_cons_ana, label='Analytical')
    plt.xlabel('Sensing range [m]')
    plt.ylabel('Mean connected vehicles')
    plt.legend(loc='best')
    plt.grid(True)

if __name__ == '__main__':
    DEBUG = True
    REPS = 1000  # Monte Carlo runs
    LAM_V = 1e-2  # (Relative) vehicle rate
    LAM_S = 1e-3  # (Relative) street rate
    ROAD_LEN = 5000  # length of roads
    # range of communication
    COM_RANGES = np.arange(1, 1000, 1)

    if DEBUG:
        time_start = time.process_time()

    # Own street
    OWN_MEAN_CONS, OWN_MEAN_CONS_ANA, OWN_COORDS_VEH, OWN_COORDS_VEH_OWN = \
        count_cons_own_street(REPS, LAM_V, ROAD_LEN, COM_RANGES)
    # Parallel streets
    PAR_MEAN_CONS, PAR_MEAN_CONS_ANA, PAR_COORDS_VEH, PAR_COORDS_VEH_OWN = \
        count_cons_par_streets(REPS, LAM_V, LAM_S, ROAD_LEN, COM_RANGES)
    # Orthogonal streets
    ORTH_MEAN_CONS, ORTH_MEAN_CONS_ANA, ORTH_COORDS_VEH, ORTH_COORDS_VEH_OWN = \
        count_cons_orth_streets(REPS, LAM_V, LAM_S, ROAD_LEN, COM_RANGES)

    if DEBUG:
        time_diff = time.process_time() - time_start
        print(time_diff)

    # Plot the results
    OWN_COORDS_VEH_OWN = np.hstack((OWN_COORDS_VEH_OWN, 0))
    OWN_COORDS_VEH = np.vstack(
        (OWN_COORDS_VEH, np.zeros_like(OWN_COORDS_VEH))).T
    plot_results(OWN_MEAN_CONS, OWN_MEAN_CONS_ANA, COM_RANGES,
                 OWN_COORDS_VEH, OWN_COORDS_VEH_OWN)
    plot_results(PAR_MEAN_CONS, PAR_MEAN_CONS_ANA, COM_RANGES,
                 PAR_COORDS_VEH, PAR_COORDS_VEH_OWN)
    plot_results(ORTH_MEAN_CONS, ORTH_MEAN_CONS_ANA, COM_RANGES,
                 ORTH_COORDS_VEH, ORTH_COORDS_VEH_OWN)
    # Show all plots
    plt.show()
