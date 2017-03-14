""" Package to simulate manhatten grid with pathloss"""

import numpy as np
from matplotlib import pyplot as plt


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
    coords_own = coords_veh[centroid_index, :]
    return coords_own


if __name__ == '__main__':
    LAM_S = 1e-3
    LAM_V = 1e-2
    ROAD_LEN = 5000
    COORDS_VEH_X = gen_streets_and_vehicles(LAM_S, LAM_V, ROAD_LEN)
    COORDS_VEH_Y = np.fliplr(gen_streets_and_vehicles(LAM_S, LAM_V, ROAD_LEN))
    COORDS_VEH = np.vstack((COORDS_VEH_X, COORDS_VEH_Y))
    COORDS_OWN = find_own_veh(ROAD_LEN, COORDS_VEH)
    plt.figure()
    plt.scatter(COORDS_VEH[:, 0], COORDS_VEH[:, 1])
    plt.scatter(COORDS_OWN[0], COORDS_OWN[1])
    plt.show()
