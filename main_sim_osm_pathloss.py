""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# TODO: DEPRECATED, main script is now main_sim_osm.py!

# Standard imports
import os
import logging

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles
import propagation as prop
import network_parser as nw_p


def main_sim_single(network, max_pl=150):
    """Simulates the connections from one to all other vehicles using pathloss functions"""

    # Initialize
    vehs = network['vehs']
    graph_streets_wave = network['graph_streets_wave']
    gdf_buildings = network['gdf_buildings']
    count_veh = vehs.count
    vehs.allocate(count_veh)

    # Find center vehicle
    time_start = utils.debug(None, 'Finding center vehicle')
    idx_center_veh = geom_o.find_center_veh(vehs.get())
    idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
    vehs.add_key('center', idx_center_veh)
    vehs.add_key('other', idxs_other_vehs)
    utils.debug(time_start)

    # Determine propagation conditions
    time_start = utils.debug(None, 'Determining propagation conditions')
    is_nlos = prop.veh_cons_are_nlos(vehs.get_points('center'),
                                     vehs.get_points('other'), gdf_buildings)
    vehs.add_key('nlos', idxs_other_vehs[is_nlos])
    is_olos_los = np.invert(is_nlos)
    vehs.add_key('olos_los', idxs_other_vehs[is_olos_los])
    utils.debug(time_start)

    # Determine OLOS and LOS
    time_start = utils.debug(None, 'Determining OLOS and LOS')
    # NOTE: A margin of 2, means round cars with radius 2 meters
    is_olos = prop.veh_cons_are_olos(vehs.get_points('center'),
                                     vehs.get_points('olos_los'), margin=2)
    is_los = np.invert(is_olos)
    vehs.add_key('olos', vehs.get_idxs('olos_los')[is_olos])
    vehs.add_key('los', vehs.get_idxs('olos_los')[is_los])
    utils.debug(time_start)

    # Determine orthogonal and parallel
    time_start = utils.debug(None, 'Determining orthogonal and parallel')

    is_orthogonal, coords_intersections = \
        prop.check_if_cons_are_orthogonal(graph_streets_wave,
                                          vehs.get_graph('center'),
                                          vehs.get_graph('nlos'),
                                          max_angle=np.pi)
    is_parallel = np.invert(is_orthogonal)
    vehs.add_key('orth', vehs.get_idxs('nlos')[is_orthogonal])
    vehs.add_key('par', vehs.get_idxs('nlos')[is_parallel])
    utils.debug(time_start)

    # Determining pathlosses for LOS and OLOS
    time_start = utils.debug(None, 'Calculating pathlosses for OLOS and LOS')

    p_loss = pathloss.Pathloss()
    distances_olos_los = np.sqrt(
        (vehs.get('olos_los')[:, 0] - vehs.get('center')[0])**2 +
        (vehs.get('olos_los')[:, 1] - vehs.get('center')[1])**2)

    pathlosses_olos = p_loss.pathloss_olos(distances_olos_los[is_olos])
    vehs.set_pathlosses('olos', pathlosses_olos)
    pathlosses_los = p_loss.pathloss_los(distances_olos_los[is_los])
    vehs.set_pathlosses('los', pathlosses_los)
    utils.debug(time_start)

    # Determining pathlosses for NLOS orthogonal
    time_start = utils.debug(
        None, 'Calculating pathlosses for NLOS orthogonal')

    # NOTE: Assumes center vehicle is receiver
    # NOTE: Uses airline vehicle -> intersection -> vehicle and not
    # street route
    distances_orth_tx = np.sqrt(
        (vehs.get('orth')[:, 0] - coords_intersections[is_orthogonal, 0])**2 +
        (vehs.get('orth')[:, 1] - coords_intersections[is_orthogonal, 1])**2)
    distances_orth_rx = np.sqrt(
        (vehs.get('center')[0] - coords_intersections[is_orthogonal, 0])**2 +
        (vehs.get('center')[1] - coords_intersections[is_orthogonal, 1])**2)
    pathlosses_orth = p_loss.pathloss_nlos(
        distances_orth_rx, distances_orth_tx)
    vehs.set_pathlosses('orth', pathlosses_orth)
    pathlosses_par = np.Infinity * np.ones(np.sum(is_parallel))
    vehs.set_pathlosses('par', pathlosses_par)
    utils.debug(time_start)

    # Determine in range / out of range
    time_start = utils.debug(None, 'Determining in range vehicles')
    idxs_in_range = vehs.get_pathlosses('other') < max_pl
    idxs_out_range = np.invert(idxs_in_range)
    vehs.add_key('in_range', vehs.get_idxs('other')[idxs_in_range])
    vehs.add_key('out_range', vehs.get_idxs('other')[idxs_out_range])
    utils.debug(time_start)


def main():
    """Main simulation function"""

    config = nw_p.params_from_conf()
    config_scenario = nw_p.params_from_conf(config['scenario'])
    config.update(config_scenario)

    if isinstance(config['densities_veh'], (list, tuple)):
        densities = np.zeros(0)
        for density_in in config['densities_veh']:
            if isinstance(density_in, dict):
                density = np.linspace(**density_in)
            else:
                density = density_in
            densities = np.append(densities, density)
        config['densities_veh'] = densities

    # Logger setup
    if 'loglevel' not in config:
        config['logelevel'] = 'ERROR'

    loglevel = logging.getLevelName(config['loglevel'])
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Setup OSMnx
    ox.config(log_console=True, log_level=loglevel, use_cache=True)

    # Switch to selected simulation mode
    if config['sim_mode'] == 'single':
        if np.size(config['densities_veh']) > 1:
            raise ValueError(
                'Single simulation mode can only simulate 1 density value')

        net = ox_a.load_network(config['place'],
                                which_result=config['which_result'])
        vehicles.place_vehicles_in_network(net, density_veh=config['densities_veh'],
                                           density_type=config['density_type'])
        main_sim_single(net, max_pl=config['max_pl'])

        if config['show_plot']:
            plot.plot_prop_cond(net['graph_streets'], net['gdf_buildings'],
                                net['vehs'], show=False)
            plot.plot_pathloss(net['graph_streets'], net['gdf_buildings'],
                               net['vehs'], show=False)
            plot.plot_con_status(net['graph_streets'], net['gdf_buildings'],
                                 net['vehs'], show=False)
            plt.show()

    else:
        raise NotImplementedError('Simulation type not supported')


if __name__ == '__main__':
    # Change to directory of script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Run main function
    main()
