""" Generates streets, buildings and vehicles from OpenStreetMap data with osmnx"""

# Standard imports
import os.path
import ipdb

# Extension imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import networkx as nx

# Local imports
import pathloss
import plot
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles
import propagation as prop


def main_sim(place, which_result=1, density_veh=100, use_pathloss=True, max_pl=150, density_type='absolute', single_veh=True,
             debug=False):
    """ Test the whole functionality"""

    # Setup
    ox_a.setup(debug)

    # Load data
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle'.format(
        utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle'.format(
        utils.string_to_filename(place))
    filename_data_boundary = 'data/{}_boundary.pickle'.format(
        utils.string_to_filename(place))

    if os.path.isfile(filename_data_streets) and os.path.isfile(filename_data_buildings) and \
            os.path.isfile(filename_data_boundary):
        # Load from file
        time_start = utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        time_start = utils.debug(debug, None, 'Loading data from the internet')
        data = ox_a.download_place(place, which_result=which_result)

    utils.debug(debug, time_start)

    # Choose random streets and position on streets
    time_start = utils.debug(
        debug, None, 'Building graph for wave propagation')

    graph_streets = data['streets']
    gdf_buildings = data['buildings']

    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    ox_a.add_geometry(graph_streets)
    streets_wave = graph_streets.to_undirected()
    prop.add_edges_if_los(streets_wave, gdf_buildings)

    utils.debug(debug, time_start)

    # Streets and positions selection
    time_start = utils.debug(debug, None, 'Choosing random vehicle positions')

    street_lengths = geom_o.get_street_lengths(graph_streets)

    if density_type == 'absolute':
        count_veh = int(density_veh)
    elif density_type == 'length':
        count_veh = int(round(density_veh * np.sum(street_lengths)))
    elif density_type == 'area':
        area = data['boundary'].area
        count_veh = int(round(density_veh * area))
    else:
        raise ValueError('Density type not supported')

    rand_street_idxs = vehicles.choose_random_streets(
        street_lengths, count_veh)
    utils.debug(debug, time_start)

    # Vehicle generation
    time_start = utils.debug(debug, None, 'Generating vehicles')
    vehs = vehicles.generate_vehs(graph_streets, rand_street_idxs)
    utils.debug(debug, time_start)

    if single_veh:
        # Find center vehicle
        time_start = utils.debug(debug, None, 'Finding center vehicle')

        idx_center_veh = geom_o.find_center_veh(vehs.get())
        idxs_other_vehs = np.where(np.arange(count_veh) != idx_center_veh)[0]
        vehs.add_key('center', idx_center_veh)
        vehs.add_key('other', idxs_other_vehs)

        utils.debug(debug, time_start)

        # Determine NLOS and OLOS/LOS
        time_start = utils.debug(
            debug, None, 'Determining propagation conditions')

        is_nlos = prop.veh_cons_are_nlos(vehs.get_points('center'),
                                         vehs.get_points('other'), gdf_buildings)
        vehs.add_key('nlos', idxs_other_vehs[is_nlos])
        is_olos_los = np.invert(is_nlos)
        vehs.add_key('olos_los', idxs_other_vehs[is_olos_los])

        utils.debug(debug, time_start)

        if use_pathloss:
            # Determine OLOS and LOS
            time_start = utils.debug(debug, None, 'Determining OLOS and LOS')

            # NOTE: A margin of 2, means round cars with radius 2 meters
            is_olos = prop.veh_cons_are_olos(vehs.get_points('center'),
                                             vehs.get_points('olos_los'), margin=2)
            is_los = np.invert(is_olos)
            vehs.add_key('olos', vehs.get_idxs('olos_los')[is_olos])
            vehs.add_key('los', vehs.get_idxs('olos_los')[is_los])

            utils.debug(debug, time_start)

            # Determine orthogonal and parallel
            time_start = utils.debug(
                debug, None, 'Determining orthogonal and parallel')

            is_orthogonal, coords_intersections = \
                prop.check_if_cons_orthogonal(streets_wave,
                                              vehs.get_graph('center'),
                                              vehs.get_graph('nlos'),
                                              max_angle=np.pi)
            is_parallel = np.invert(is_orthogonal)
            vehs.add_key('orth', vehs.get_idxs('nlos')[is_orthogonal])
            vehs.add_key('par', vehs.get_idxs('nlos')[is_parallel])

            utils.debug(debug, time_start)

            # Determining pathlosses for LOS and OLOS
            time_start = utils.debug(
                debug, None, 'Calculating pathlosses for OLOS and LOS')

            p_loss = pathloss.Pathloss()
            distances_olos_los = np.sqrt(
                (vehs.get('olos_los')[:, 0] - vehs.get('center')[0])**2 +
                (vehs.get('olos_los')[:, 1] - vehs.get('center')[1])**2)

            # TODO: why - ? fix in pathloss.py
            pathlosses_olos = - \
                p_loss.pathloss_olos(distances_olos_los[is_olos])
            vehs.set_pathlosses('olos', pathlosses_olos)
            # TODO: why - ? fix in pathloss.py
            pathlosses_los = -p_loss.pathloss_los(distances_olos_los[is_los])
            vehs.set_pathlosses('los', pathlosses_los)

            utils.debug(debug, time_start)

            # Determining pathlosses for NLOS orthogonal
            time_start = utils.debug(
                debug, None, 'Calculating pathlosses for NLOS orthogonal')

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

            utils.debug(debug, time_start)

            # Determine in range / out of range
            time_start = utils.debug(
                debug, None, 'Determining in range vehicles')

            idxs_in_range = vehs.get_pathlosses('other') < max_pl
            idxs_out_range = np.invert(idxs_in_range)
            vehs.add_key('in_range', vehs.get_idxs('other')[idxs_in_range])
            vehs.add_key('out_range', vehs.get_idxs('other')[idxs_out_range])

            utils.debug(debug, time_start)

            # Plots specific to pathloss method
            plot.plot_prop_cond(graph_streets, gdf_buildings,
                                vehs, show=False, place=place)
            plot.plot_pathloss(graph_streets, gdf_buildings,
                               vehs, show=False, place=place)

        else:  # not use_pathloss
            time_start = utils.debug(
                debug, None, 'Determining in range vehicles')

            # TODO: make parameters
            max_dist_olos_los = 250
            max_dist_nlos = 140
            distances = np.linalg.norm(
                vehs.get('other') - vehs.get('center'), ord=2, axis=1)
            vehs.set_distances('other', distances)

            is_in_range_olos_los = vehs.get_distances(
                'olos_los') < max_dist_olos_los
            idxs_in_range_olos_los = vehs.get_idxs(
                'olos_los')[is_in_range_olos_los]

            is_in_range_nlos = vehs.get_distances('nlos') < max_dist_nlos
            idxs_in_range_nlos = vehs.get_idxs('nlos')[is_in_range_nlos]
            idxs_in_range = np.append(
                idxs_in_range_olos_los, idxs_in_range_nlos)
            vehs.add_key('in_range', idxs_in_range)

            idxs_out_range = vehs.get_idxs('other')\
                [np.where(np.invert(np.in1d(vehs.get_idxs('other'), idxs_in_range)))]
            vehs.add_key('out_range', idxs_out_range)

            utils.debug(debug, time_start)

        # Common plots for both methods
        plot.plot_con_status(graph_streets, gdf_buildings,
                             vehs, show=False, place=place)
        plt.show()

    else:  # not single_veh
        if use_pathloss:
            raise NotImplementedError('Not yet supported')
        else:  # not use_pathloss
            # TODO: make parameters
            max_dist_olos_los = 250
            max_dist_nlos = 140
            max_dist = max(max_dist_nlos, max_dist_olos_los)
            # Determine NLOS and OLOS/LOS
            time_start = utils.debug(
                debug, None, 'Determining propagation conditions')
            # vehs.is_nlos = ... TODO
            is_nlos = prop.veh_cons_are_nlos_all(
                vehs.get_points(), gdf_buildings, max_dist=max_dist)
            is_olos_los = np.invert(is_nlos)
            idxs_olos_los = np.where(is_olos_los)[0]
            idxs_nlos = np.where(is_nlos)[0]

            count_cond = count_veh * (count_veh - 1) // 2

            utils.debug(debug, time_start)

            time_start = utils.debug(
                debug, None, 'Determining in range vehicles')

            distances = dist.pdist(vehs.coordinates)
            idxs_in_range_olos_los = idxs_olos_los[
                distances[idxs_olos_los] < max_dist_olos_los]
            idxs_in_range_nlos = idxs_nlos[
                distances[idxs_nlos] < max_dist_nlos]
            idxs_in_range = np.append(
                idxs_in_range_olos_los, idxs_in_range_nlos)
            is_in_range = np.in1d(np.arange(count_cond), idxs_in_range)
            is_in_range_matrix = dist.squareform(is_in_range).astype(bool)
            graph_cons = nx.from_numpy_matrix(is_in_range_matrix)

            clusters = nx.connected_component_subgraphs(graph_cons)
            # TODO: iterate clusters to find max
            cluster_max = max(clusters, key=len)
            net_connectivity = cluster_max.order() / count_veh
            vehs.add_key('cluster_max', cluster_max.nodes())
            not_cluster_max_nodes = np.arange(count_veh)[~np.in1d(
                np.arange(count_veh), cluster_max.nodes())]
            vehs.add_key('not_cluster_max', not_cluster_max_nodes)

            utils.debug(debug, time_start)
            print('Network connectivity: {:.2f} %'.format(
                net_connectivity * 100))

            plot.plot_cluster_max(graph_streets, gdf_buildings,
                                  vehs, show=False, place=place)
            plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    main_sim(args.p, which_result=args.w, density_veh=args.d, use_pathloss=bool(args.s), max_pl=args.l, density_type=args.t, single_veh=bool(args.v),
             debug=True)
