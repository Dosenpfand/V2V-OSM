""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import time
import os.path
import multiprocessing as mp
from itertools import repeat
import pickle
from itertools import repeat

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import networkx as nx
import json
# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles as vehicle_manager
import propagation as prop
import shapely.geometry as geo



def main_sim(network, max_pl=150, debug=False, max_dist_olos_los=300, center_veh=0):
    """Simulates the connections from one to all other vehicles using pathloss functions """
    idx = center_veh
    # Initialize
    vehs = network['vehs']
    graph_streets_wave = network['graph_streets_wave']
    gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    vehs.allocate(count_veh)
    # Find center vehicle
    # time_start = # utils.debug(debug, None, 'Finding center vehicle')

    print("\t Current Run: {}/{}".format(idx, count_veh))
    idx_center_veh = center_veh
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    # utils.debug(debug, # time_start)

    # Determine propagation conditions
    # time_start = # utils.debug(
    #    debug, None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos(vehs.get_points('center'),
                                     vehs.get_points('other'),
                                     gdf_buildings,
                                     max_dist=max_dist_olos_los)
    vehs.add_key('nlos', idxs_other_vehs[is_nlos])
    is_olos_los = np.invert(is_nlos)
    vehs.add_key('olos_los', idxs_other_vehs[is_olos_los])
    # utils.debug(debug, # time_start)

    # Determine OLOS and LOS
    # time_start = # utils.debug(debug, None, 'Determining OLOS and LOS')
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos = prop.veh_cons_are_olos(vehs.get_points('center'),
                                     vehs.get_points('olos_los'), margin=2)
    is_los = np.invert(is_olos)
    vehs.add_key('olos', vehs.get_idxs('olos_los')[is_olos])
    vehs.add_key('los', vehs.get_idxs('olos_los')[is_los])
    # utils.debug(debug, # time_start)
    d = min([vehs.get_points('center').distance(geo.Point(x[1]['x'],
                                                          x[1]['y']))
             for x in graph_streets_wave.node.items()])
    return (len(vehs.get_idxs("olos_los"))), (d)


def execute_main():
    sim_mode = 'single'  # 'single', 'multi', 'multiprocess'
    place = 'Upper West Side - New York - USA'
    place = 'Linz Stadt - Austria'
    # place='Neubau - Vienna - Austria'
    which_result = 1
    densities_veh = 50e-3
    density_type = 'area'
    max_dist_olos_los = 350
    max_dist_nlos = 140
    iterations = 1
    max_pl = 150
    show_plot = False
    send_mail = False
    mail_to = 'thomas.blazek@nt.tuwien.ac.at'

    # TODO: temp!
    # place = 'Neubau - Wien - Austria'
    densities_veh = 250e-6
    iterations = 1
    # place = 'Neubau - Vienna - Austria'
    sim_mode = 'single'
    with open("network_definition.json") as fp:
        param_set = json.load(fp)
    # Adapt static input parameters
    static_params = param_set["graz"]
    # Switch to selected simulation mode
    print(sim_mode)
    if sim_mode == 'multi':
        net_connectivities = np.zeros([iterations, np.size(densities_veh)])
        for iteration in np.arange(iterations):
            for idx_density, density in enumerate(densities_veh):
                print('Densitiy: {:.2E}, Iteration: {:d}'.format(
                    density, iteration))
                net = load_network(static_params['place'],
                                   which_result=static_params['which_result'],
                                   debug=False)
                generate_vehicles(net, density_veh=density,
                                  density_type=static_params['density_type'],
                                  debug=False)
                net_connectivity, path_redundancy = main_sim_multi(net, max_dist_olos_los=max_dist_olos_los,
                                                                   max_dist_nlos=max_dist_nlos, debug=True)
                net_connectivities[iteration, idx_density] = net_connectivity
            np.save('results/net_connectivities',
                    net_connectivities[:iteration + 1])

        if show_plot:
            plot.plot_cluster_max(net['graph_streets'], net['gdf_buildings'],
                                  net['vehs'], show=False, place=place)
            plt.show()
    elif sim_mode == 'single':
        if np.size(densities_veh) > 1:
            raise ValueError(
                'Single simulation mode can only simulate 1 density value')

        net = load_network(static_params['place'],
                           which_result=static_params['which_result'],
                           debug=False)
        generate_vehicles(net, density_veh=densities_veh,
                          density_type=static_params['density_type'],
                          debug=False)
        with mp.Pool(12) as p:
            return_val = p.starmap(main_sim, zip(repeat(net), repeat(max_pl),
                                                 repeat(True), repeat(300), range(net['vehs'].count)))
        with open("return_vals.p", "wb") as fp:
            pickle.dump(return_val, fp)

    else:
        raise NotImplementedError('Simulation type not supported')
if __name__ == '__main__':
    execute_main()
