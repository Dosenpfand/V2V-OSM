"""network_parser provides configuration based network generation.

It contains the load_network function, which loads from cache
(or from OpenStreetMap) a street layout. To make this easily accessible,
it provides 2 additional functions:
network_from_conf(in_key="graz", config_file="network_definition.json")
params_from_conf(in_key="global_config", config_file="network_definition.json")

params_from_conf takes a json file, and loads a given key from that file.
network_from_conf takes that function and uses it to load a setup.
The default filename is network_definition.json.
"""

import json

import numpy as np

import osmnx_addons as ox_a


def network_from_conf(in_key="default", config_file="network_definition.json"):
    """Load a network from the settings in a json file.

    Abstracts away load_network call.
    """
    conf = params_from_conf(in_key, config_file)
    return ox_a.load_network(conf["place"], conf["which_result"])


def params_from_conf(in_key="global",
                     config_file="network_definition.json"):
    """Load a parameter set from the given config_file.

    global_config: configuration params that are independent on chosen network.
    otherwise: network paramters for load_network and range and stuff."""
    with open(config_file, "r") as file_pointer:
        conf = json.load(file_pointer)
    return conf[in_key]


def check_fill_config(config):
    """Checks mandatory settings and sets unset SUMO settings to defaults"""

    # Mandatory settings
    if 'scenario' not in config:
        raise KeyError('Scenario not set')

    if 'place' not in config:
        raise KeyError('Place not set')

    if 'distribution_veh' not in config:
        raise KeyError('Distrubution type not set')
    else:
        if config['distribution_veh'] not in ['SUMO', 'uniform']:
            raise KeyError('Vehicle distribution method not supported')

        if config['distribution_veh'] == 'uniform' and config['simulation_mode'] != 'demo':
            if 'iterations' not in config:
                raise KeyError('Number of iterations not set')

    if config['simulation_mode'] == 'demo':
        if isinstance(config['densities_veh'], (list, tuple)):
            raise KeyError('Only a single density supported in demo mode')

        if config['distribution_veh'] != 'uniform':
            raise KeyError('Only uniform vehicle distribution supported in demo mode')

        if config['connection_metric'] != 'pathloss':
            raise KeyError('Only pathloss as connection metric supported in demo mode')

    if 'densities_veh' not in config:
        raise KeyError('Vehicle densities not set')

    if 'connection_metric' not in config:
        raise KeyError('Connection metric not set')

    if 'max_connection_metric' not in config:
        raise KeyError('Maximum connection metric not set')

    if 'simulation_mode' not in config:
        raise KeyError('Simulation mode not set')

    # Optional settings
    if 'send_mail' not in config:
        config['send_mail'] = False
    else:
        if config['send_mail']:
            if 'mail_to' not in config:
                raise KeyError('Email address not set')

    if 'show_plot' not in config:
        config['show_plot'] = False

    if 'loglevel' not in config:
        config['loglevel'] = 'INFO'

    if 'which_result' not in config:
        config['which_result'] = None

    if 'building_tolerance' not in config:
        config['building_tolerance'] = 0

    # Optional SUMO settings
    if config['distribution_veh'] == 'SUMO':
        if 'sumo' not in config:
            config['sumo'] = {}
        if 'tls_settings' not in config['sumo']:
            config['sumo']['tls_settings'] = None
        if 'fringe_factor' not in config['sumo']:
            config['sumo']['fringe_factor'] = None
        if 'max_speed' not in config['sumo']:
            config['sumo']['max_speed'] = None
        if 'intermediate_points' not in config['sumo']:
            config['sumo']['intermediate_points'] = None
        if 'warmup_duration' not in config['sumo']:
            config['sumo']['warmup_duration'] = None
        if 'abort_after_sumo' not in config['sumo']:
            config['sumo']['abort_after_sumo'] = False

    # Convert densities
    config['densities_veh'] = convert_densities(config['densities_veh'])

    return config


def convert_densities(config_densities):
    """Converts the density parameters from the configuration to a simple array"""

    if isinstance(config_densities, (list, tuple)):
        densities = np.zeros(0)
        for density_in in config_densities:
            if isinstance(density_in, dict):
                density = np.linspace(**density_in)
            else:
                density = density_in
            densities = np.append(densities, density)
    else:
        densities = np.array([config_densities])

    return densities
