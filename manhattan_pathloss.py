""" Package to simulate manhatten grid with pathloss"""

import numpy as np
from matplotlib import pyplot as plt
import pathloss

def gen_streets_and_vehicles(lam_s, lam_v, road_len):
    """Generates streets and vehicles on it in 1 dimension"""
    # Truncated poisson variable realization
    count_streets = 0
    while count_streets == 0:
        count_streets = np.random.poisson(lam_s * road_len, 1)
    coords_streets = np.random.uniform(0, road_len, count_streets)

    # Truncated poisson vector realization
    counts_veh = np.zeros(count_streets, dtype=int)
    for i_street in np.arange(count_streets):
        while counts_veh[i_street] == 0:
            counts_veh[i_street] = np.random.poisson(lam_s * road_len, 1)
    count_veh_all = np.sum(counts_veh)
    coords_veh_x = np.random.uniform(0, road_len, count_veh_all)
    coords_veh_y = np.repeat(coords_streets, counts_veh)
    coords_veh = np.vstack((coords_veh_x, coords_veh_y)).T
    return coords_veh


def find_own_veh(road_len, coords_veh):
    """Searches for the coordinates of the vehicle most in the center"""
    coords_center = np.array((road_len / 2, road_len / 2))
    distances_center = np.linalg.norm(
        coords_center - coords_veh, ord=2, axis=1)
    centroid_index = np.argmin(distances_center)
    return centroid_index

def find_same_street_veh(coords_veh, coords_own):
    """Searches for vehicles on the same street as our own vehicle"""
    dir_own = int(coords_own[2])
    coords_street_own = coords_own[dir_own]
    ind_veh_same_dir = np.flatnonzero(coords_veh[:, 2] == dir_own)
    ind_veh_same_coord_dir = np.flatnonzero(coords_veh[:, dir_own] == coords_street_own)
    ind_veh_same_street = np.intersect1d(ind_veh_same_dir, ind_veh_same_coord_dir)
    return ind_veh_same_street


# def find_olos_veh(coords_same_street, coords_own):
#     dir_own = int(coords_own[2])


if __name__ == '__main__':
    # configuration
    LAM_S = 3e-3
    LAM_V = 1e-2
    ROAD_LEN = 1e4
    PL_THR = -110

    # Street and vehicle generation
    COORDS_VEH_X = gen_streets_and_vehicles(LAM_S, LAM_V, ROAD_LEN)
    COORDS_VEH_X = np.column_stack((COORDS_VEH_X, np.ones(np.shape(COORDS_VEH_X)[0])))
    COORDS_VEH_Y = np.fliplr(gen_streets_and_vehicles(LAM_S, LAM_V, ROAD_LEN))
    COORDS_VEH_Y = np.column_stack((COORDS_VEH_Y, np.zeros(np.shape(COORDS_VEH_Y)[0])))
    COORDS_VEH = np.vstack((COORDS_VEH_X, COORDS_VEH_Y))
    INDEX_OWN = find_own_veh(ROAD_LEN, COORDS_VEH[:, 0:2])
    COORDS_OWN = COORDS_VEH[INDEX_OWN, :]
    COORDS_VEH = np.delete(COORDS_VEH, INDEX_OWN, axis=0)

    # Pathloss calculation
    DISTANCES = np.linalg.norm(COORDS_VEH[:, 0:2] - COORDS_OWN[0:2], ord=2, axis=1)
    PL = pathloss.Pathloss()
    PATHLOSSES_LOS = PL.pathloss_los(DISTANCES)
    COORDS_IN_RANGE_IND = np.flatnonzero(PATHLOSSES_LOS > PL_THR)
    COORDS_IN_RANGE = COORDS_VEH[COORDS_IN_RANGE_IND, :]
    COORDS_NOT_IN_RANGE = np.delete(COORDS_VEH, COORDS_IN_RANGE_IND, axis=0)

    # Plot same street vehicles
    IND_SAME_STREET = find_same_street_veh(COORDS_VEH, COORDS_OWN)
    COORDS_SAME_STREET = COORDS_VEH[IND_SAME_STREET, :]
    COORDS_NOT_SAME_STREET = np.delete(COORDS_VEH, IND_SAME_STREET, axis=0)
    plt.figure()
    plt.scatter(COORDS_NOT_SAME_STREET[:, 0], COORDS_NOT_SAME_STREET[:, 1],
                label='Not same street')
    plt.scatter(COORDS_SAME_STREET[:, 0], COORDS_SAME_STREET[:, 1],
                label='Same street')
    plt.scatter(COORDS_OWN[0], COORDS_OWN[1], label='Own vehicle')
    plt.legend()
    plt.grid(True)

    # Plot x and y vehicles
    COORDS_X_IND = np.flatnonzero(COORDS_VEH[:, 2] == 1)
    COORDS_Y_IND = np.flatnonzero(COORDS_VEH[:, 2] == 0)
    plt.figure()
    plt.scatter(COORDS_VEH[COORDS_X_IND, 0], COORDS_VEH[COORDS_X_IND, 1],
                label='Traveling in x direction')
    plt.scatter(COORDS_VEH[COORDS_Y_IND, 0], COORDS_VEH[COORDS_Y_IND, 1],
                label='Traveling in y direction')
    plt.legend()
    plt.grid(True)

    # Plot vehicles in range
    plt.figure()
    plt.scatter(COORDS_NOT_IN_RANGE[:, 0], COORDS_NOT_IN_RANGE[:, 1], label='Not in range')
    plt.scatter(COORDS_IN_RANGE[:, 0], COORDS_IN_RANGE[:, 1], label='In range')
    plt.scatter(COORDS_OWN[0], COORDS_OWN[1], label='Own vehicle')
    plt.legend()
    plt.grid(True)
    plt.show()


