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
            counts_veh[i_street] = np.random.poisson(lam_v * road_len, 1)
    count_veh_all = np.sum(counts_veh)
    coords_veh_x = np.random.uniform(0, road_len, count_veh_all)
    coords_veh_y = np.repeat(coords_streets, counts_veh)
    coords_veh = np.vstack((coords_veh_x, coords_veh_y)).T
    return coords_veh, coords_streets


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


def find_los_veh(coords_same_street, coords_own):
    """Finds the vehicles that have a line of sight to the own vehicle"""
    dir_own = int(coords_own[2])
    ind_diff = 1-dir_own
    distances = coords_own[ind_diff] - coords_same_street[:, ind_diff]
    ind_distances_pos = np.flatnonzero(distances > 0)
    distances_pos = distances[ind_distances_pos]
    ind_los_pos = np.argmin(distances_pos)
    ind_distances_neg = np.flatnonzero(distances < 0)
    distances_neg = distances[ind_distances_neg]
    ind_los_neg = np.argmax(distances_neg)
    return ind_distances_pos[ind_los_pos], ind_distances_neg[ind_los_neg]

def find_nlos_veh(coords_other_streets, coords_own):
    """Finds vehicles that have non line of sight to the own vehicle"""
    dir_own = int(coords_own[2])
    ind_nlos = np.flatnonzero(coords_other_streets[:, 2] != dir_own)
    return ind_nlos

def count_connections(distances, distances_max):
    """Determines how many distances are smaller than the elements in distances_max"""
    distances_max_tiled = np.tile(distances_max, (distances.size, 1)).T
    distances_tiled = np.tile(distances, (distances_max.size, 1))
    counts_range = np.count_nonzero(distances_tiled < distances_max_tiled, axis=1)
    return counts_range

def plot_cons_vs_max_pl(pl_thr_range, cons_mean):
    """Plots connections vs pathloss threshold"""

    # Plot
    plt.figure()
    plt.plot(pl_thr_range, cons_mean)
    plt.grid(True)

    # Add text
    plt.title('Connections')
    plt.xlabel('Max. pathloss [dB]')
    plt.ylabel('Number of connections')


def plot_scatter_range(coords_own, coords_in_range, coords_out_range, coords_streets_x,
                       coords_streets_y, road_len):
    """Scatter plot of the vehicles in and out of range"""

    #Prepare
    plt.figure()

    # Plot streets
    for coords_street_x in coords_streets_x:
        plt.plot((0, road_len), (coords_street_x, coords_street_x), c='grey', linewidth=0.5, \
            alpha=0.5)
    for coords_street_y in coords_streets_y:
        plt.plot((coords_street_y, coords_street_y), (0, road_len), c='grey', linewidth=0.5, \
            alpha=0.5)

    # Plot vehicles
    plt.scatter(coords_in_range[:, 0], coords_in_range[:, 1], marker='x', label='In range')
    plt.scatter(coords_out_range[:, 0], coords_out_range[:, 1], marker='x', label='Out of range')
    plt.scatter(coords_own[0], coords_own[1], c='black', marker='o', label='Own vehicle')

    # Add text
    plt.title('Vehicle positions and range')
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()

def plot_scatter_pathloss(coords_own, coords_veh, pathlosses, coords_streets_x, coords_streets_y, \
                          road_len):
    """Scatter plot of the vehicles with color according to pathloss"""

    # Prepare
    fig, axi = plt.subplots()

    # Plot streets
    for coords_street_x in coords_streets_x:
        plt.plot((0, road_len), (coords_street_x, coords_street_x), c='grey', linewidth=0.5, \
            alpha=0.5)
    for coords_street_y in coords_streets_y:
        plt.plot((coords_street_y, coords_street_y), (0, road_len), c='grey', linewidth=0.5, \
            alpha=0.5)

    # Plot vehicles
    plt.scatter(coords_own[0], coords_own[1], c='black', marker='o', label='Own vehicle')
    cax = plt.scatter(coords_veh[:, 0], coords_veh[:, 1], marker='x', c=pathlosses, \
        cmap=plt.cm.magma, label='Other vehicles')

    # Add text to plot
    axi.set_title('Vehicle positions and pathloss')
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()

    # Add colorbar for pathloss
    pl_min = np.min(pathlosses)
    pl_max = np.max(pathlosses)
    pl_med = np.mean((pl_min, pl_max))
    string_min = '{:.0f}'.format(pl_min)
    string_med = '{:.0f}'.format(pl_med)
    string_max = '{:.0f}'.format(pl_max)
    cbar = fig.colorbar(cax, ticks=[pl_min, pl_med, pl_max], orientation='vertical')
    cbar.ax.set_xticklabels([string_min, string_med, string_max])
    cbar.ax.set_xlabel('Pathloss [dB]')


def plot_scatter_propagation_cond(coords_own, coords_los, coords_olos, coords_nlos,
                                  coords_very_high_pl, coords_streets_x, coords_streets_y, \
                                  road_len):
    """Scatter plot of the vehicles with colors according to the propagation condition"""

    #Prepare
    plt.figure()

    # Plot streets
    for coords_street_x in coords_streets_x:
        plt.plot((0, road_len), (coords_street_x, coords_street_x), c='grey', linewidth=0.5, \
            alpha=0.5)
    for coords_street_y in coords_streets_y:
        plt.plot((coords_street_y, coords_street_y), (0, road_len), c='grey', linewidth=0.5, \
            alpha=0.5)

    # Plot vehicles
    plt.scatter(coords_very_high_pl[:, 0], coords_very_high_pl[:, 1], marker='x', \
        label='Very high PL')
    plt.scatter(coords_nlos[:, 0], coords_nlos[:, 1], marker='x', label='NLOS')
    plt.scatter(coords_olos[:, 0], coords_olos[:, 1], marker='x', label='OLOS')
    plt.scatter(coords_los[:, 0], coords_los[:, 1], marker='x', label='LOS')
    plt.scatter(coords_own[0], coords_own[1], c='black', marker='o', label='Own vehicle')

    # Add text
    plt.title('Vehicle positions and propagation conditions')
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()

def main_sim():
    """Main simulation function"""
    # configuration
    lam_s = 3e-3
    lam_v = 2e-2
    road_len = 5e3
    pl_thr_range = np.arange(50, 250, 2)
    reps = int(1e3)

    cons = np.zeros((np.size(pl_thr_range), reps))

    for i_rep in np.arange(reps):
        # Debug info
        print(i_rep)

        # Street and vehicle generation
        coords_veh_x, coords_streets_x = gen_streets_and_vehicles(lam_s, lam_v, road_len)
        coords_veh_x = np.column_stack((coords_veh_x, np.ones(np.shape(coords_veh_x)[0])))
        coords_veh_y, coords_streets_y = gen_streets_and_vehicles(lam_s, lam_v, road_len)
        coords_veh_y = np.fliplr(coords_veh_y)
        coords_veh_y = np.column_stack((coords_veh_y, np.zeros(np.shape(coords_veh_y)[0])))
        coords_veh = np.vstack((coords_veh_x, coords_veh_y))
        ind_own = find_own_veh(road_len, coords_veh[:, 0:2])
        coords_own = coords_veh[ind_own, :]
        coords_veh = np.delete(coords_veh, ind_own, axis=0)

        # Determine LOS/OLOS/NLOS
        ind_same_street = find_same_street_veh(coords_veh, coords_own)
        ind_other_streets = np.setdiff1d(np.arange(np.shape(coords_veh)[0]), ind_same_street)
        coords_same_street = coords_veh[ind_same_street, :]
        coords_other_streets = coords_veh[ind_other_streets, :]
        ind_los = np.asarray(find_los_veh(coords_same_street, coords_own))
        ind_olos = np.setdiff1d(np.arange(np.shape(coords_same_street)[0]), ind_los)
        coords_los = coords_same_street[ind_los, :]
        coords_olos = coords_same_street[ind_olos, :]
        ind_nlos = find_nlos_veh(coords_other_streets, coords_own)
        ind_very_high_pl = np.setdiff1d(np.arange(np.shape(coords_other_streets)[0]), ind_nlos)
        coords_nlos = coords_other_streets[ind_nlos, :]
        coords_very_high_pl = coords_other_streets[ind_very_high_pl, :]

        # Pathloss calculation
        # TODO optimize
        distances_same_street = np.linalg.norm(coords_same_street[:, 0:2] - coords_own[0:2],
                                               ord=2, axis=1)
        # TODO: own vehicle assumed to be receiver
        dir_own = int(coords_own[2])
        distances_nlos_rx = np.abs(coords_nlos[:, 1-dir_own] - coords_own[1-dir_own])
        distances_nlos_tx = np.abs(coords_nlos[:, dir_own] - coords_own[dir_own])
        pl_obj = pathloss.Pathloss()

        # TODO: minus for both functions?
        pathlosses_los = -pl_obj.pathloss_los(distances_same_street[ind_los])
        pathlosses_olos = -pl_obj.pathloss_olos(distances_same_street[ind_olos])
        pathlosses_nlos = pl_obj.pathloss_nlos(distances_nlos_rx, distances_nlos_tx)
        # TODO: infinity?
        # pathlosses_very_high_pl = np.inf*np.ones_like(ind_very_high_pl)
        pathlosses_very_high_pl = 450*np.ones_like(ind_very_high_pl)

        pathlosses = np.zeros(np.shape(coords_veh)[0])
        pathlosses[ind_same_street[ind_los]] = pathlosses_los
        pathlosses[ind_same_street[ind_olos]] = pathlosses_olos
        pathlosses[ind_other_streets[ind_nlos]] = pathlosses_nlos
        pathlosses[ind_other_streets[ind_very_high_pl]] = pathlosses_very_high_pl

        # Count connections for all max pathlosses
        cons[:, i_rep] = count_connections(pathlosses, pl_thr_range)

    # Determine mean number of connections
    cons_mean = np.mean(cons, axis=1)

    # Save for plotting
    ind_in_range = np.flatnonzero(pathlosses < pl_thr_range[-1])
    coords_in_range = coords_veh[ind_in_range, :]
    coords_out_range = np.delete(coords_veh, ind_in_range, axis=0)

    # Plots
    plot_cons_vs_max_pl(pl_thr_range, cons_mean)
    plot_scatter_range(coords_own, coords_in_range, coords_out_range, coords_streets_x,
                       coords_streets_y, road_len)
    plot_scatter_range(coords_own, coords_in_range, coords_out_range, coords_streets_x,
                       coords_streets_y, road_len)
    plot_scatter_propagation_cond(coords_own, coords_los, coords_olos, coords_nlos,
                                  coords_very_high_pl, coords_streets_x, coords_streets_y,
                                  road_len)
    plt.show()

if __name__ == '__main__':
    main_sim()
