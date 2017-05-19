"""Demonstration module mainly suited for generating plots"""

import numpy as np

import geometry as geom_o
import pathloss
import propagation as prop
import utils


def simulate(network, max_pl=150):
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
    vehs.add_key('ort', vehs.get_idxs('nlos')[is_orthogonal])
    vehs.add_key('par', vehs.get_idxs('nlos')[is_parallel])
    utils.debug(time_start)

    # Determining pathlosses for LOS and OLOS
    time_start = utils.debug(None, 'Calculating pathlosses for OLOS and LOS')

    p_loss = pathloss.Pathloss()
    distances_olos_los = np.sqrt(
        (vehs.get('olos_los')[:, 0] - vehs.get('center')[0]) ** 2 +
        (vehs.get('olos_los')[:, 1] - vehs.get('center')[1]) ** 2)

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
        (vehs.get('ort')[:, 0] - coords_intersections[is_orthogonal, 0]) ** 2 +
        (vehs.get('ort')[:, 1] - coords_intersections[is_orthogonal, 1]) ** 2)
    distances_orth_rx = np.sqrt(
        (vehs.get('center')[0] - coords_intersections[is_orthogonal, 0]) ** 2 +
        (vehs.get('center')[1] - coords_intersections[is_orthogonal, 1]) ** 2)
    pathlosses_orth = p_loss.pathloss_nlos(
        distances_orth_rx, distances_orth_tx)
    vehs.set_pathlosses('ort', pathlosses_orth)
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
