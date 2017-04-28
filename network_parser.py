"""network_parser provides configuration based network generation.

It contains the load_network function, which loads from cache
(or from openstreetmap) a street layout. To make this easily accessible,
it provides 2 additional functions:
network_from_conf(in_key="graz", config_file="network_definition.json")
params_from_conf(in_key="global_config",
                 config_file="network_definition.json")

params_from_conf takes a json file, and loads a given key from that file.
network_from_conf takes that function and uses it to load a setup.
The default filename is network_definition.json.
TODO: Similar wrapper for gen_vehicles
"""

import os.path
import pickle
import json
# Extension imports
import numpy as np

# Local imports
import utils
import osmnx_addons as ox_a
import geometry as geom_o
import vehicles as vehicle_manager
import propagation as prop


def load_network(place, which_result=1, debug=False):
    """Generates streets and buildings and"""

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
    filename_data_wave = 'data/{}_wave.pickle'.format(
        utils.string_to_filename(place))

    if os.path.isfile(filename_data_streets) and \
        os.path.isfile(filename_data_buildings) and \
            os.path.isfile(filename_data_boundary):
        # Load from file
        # time_start = # utils.debug(debug, None, 'Loading data from disk')
        data = ox_a.load_place(file_prefix)
    else:
        # Load from internet
        # time_start = # utils.debug(debug, None, 'Loading data from the
        # internet')
        data = ox_a.download_place(place, which_result=which_result)

    graph_streets = data['streets']
    gdf_buildings = data['buildings']
    gdf_boundary = data['boundary']
    ox_a.add_geometry(graph_streets)

    # utils.debug(debug, # time_start)

    # Generate wave propagation graph:
    # Vehicles are placed in a undirected version of
    # the graph because electromagnetic
    # waves do not respect driving directions
    if os.path.isfile(filename_data_wave):
        # Load from file
        # time_start = # utils.debug(
        #    debug, None, 'Loading graph for wave propagation')
        with open(filename_data_wave, 'rb') as file:
            graph_streets_wave = pickle.load(file)
    else:
        # Generate
        # time_start = # utils.debug(
        #    debug, None, 'Generating graph for wave propagation')
        graph_streets_wave = graph_streets.to_undirected()
        # TODO: check if add_edges_if_los() is really working!!!
        prop.add_edges_if_los(graph_streets_wave, gdf_buildings)
        with open(filename_data_wave, 'wb') as file:
            pickle.dump(graph_streets_wave, file)

    # utils.debug(debug, # time_start)

    network = {'graph_streets': graph_streets,
               'graph_streets_wave': graph_streets_wave,
               'gdf_buildings': gdf_buildings,
               'gdf_boundary': gdf_boundary}

    return network


def generate_vehicles(network, density_veh=100,
                      density_type='absolute'):
    """Generates vehicles in the network"""

    graph_streets = network['graph_streets']

    # Streets and positions selection
    # time_start = # utils.debug(debug, None, 'Choosing random vehicle
    # positions')

    street_lengths = geom_o.get_street_lengths(graph_streets)

    if density_type == 'absolute':
        count_veh = int(density_veh)
    elif density_type == 'length':
        count_veh = int(round(density_veh * np.sum(street_lengths)))
    elif density_type == 'area':
        area = network['gdf_boundary'].area
        count_veh = int(round(density_veh * area))
    else:
        raise ValueError('Density type not supported')

    rand_street_idxs = vehicle_manager.choose_random_streets(
        street_lengths, count_veh)
    # utils.debug(debug, # time_start)

    # Vehicle generation
    # time_start = # utils.debug(debug, None, 'Generating vehicles')
    vehs = vehicle_manager.generate_vehs(graph_streets, rand_street_idxs)
    # utils.debug(debug, # time_start)

    network['vehs'] = vehs
    return vehs


def network_from_conf(in_key="graz", config_file="network_definition.json"):
    """Load a network from the settings in a json file.

    Abstracts away load_network call.
    """
    conf = params_from_conf(in_key, config_file)
    return load_network(conf["place"], conf["which_result"], True)


def params_from_conf(in_key="global_config",
                     config_file="network_definition.json"):
    """Load a parameter set from the given config_file.

    global_config: configuration params that are independent on chosen network.
    otherwise: network paramters for load_network and range and stuff."""
    with open(config_file, "r") as file_pointer:
        conf = json.load(file_pointer)
    return conf[in_key]
