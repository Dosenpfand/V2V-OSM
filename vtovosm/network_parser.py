"""Provides configuration based network generation"""

import json
import os

import numpy as np

from . import osmnx_addons as ox_a

MODULE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG_DIR = os.path.join(MODULE_PATH, 'simulations', 'network_config')
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, 'default.json')


def network_from_conf(in_key="default", config_file=DEFAULT_CONFIG_PATH):
    """Load a network from the settings in a json file.

    Abstracts away load_network call.
    """
    conf = params_from_conf(in_key, config_file)
    return ox_a.load_network(conf["place"], conf["which_result"])


def params_from_conf(in_key="global", config_file=DEFAULT_CONFIG_PATH):
    """Load a parameter set from the given config_file.

    global_config: configuration params that are independent on chosen network.
    otherwise: network paramters for load_network and range and stuff."""
    with open(config_file, "r") as file_pointer:
        conf = json.load(file_pointer)
    return conf[in_key]


def get_scenarios_list(config_file=DEFAULT_CONFIG_PATH):
    """Returns a list of scenarios that are defined in the JSON config"""

    with open(config_file, 'r') as file:
        config = json.load(file)

    scenarios = list(config.keys())
    scenarios.remove('global')

    return scenarios


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
    if 'overwrite_result' not in config:
        config['overwrite_result'] = False

    if 'send_mail' not in config:
        config['send_mail'] = False
    else:
        if config['send_mail']:
            if 'mail_to' not in config:
                raise KeyError('Email address not set')

    if 'save_plot' not in config:
        config['save_plot'] = False

    if config['save_plot']:
        if 'plot_dir' not in config:
            config['plot_dir'] = None

    if 'loglevel' not in config:
        config['loglevel'] = 'INFO'

    if 'which_result' not in config:
        config['which_result'] = None

    if 'building_tolerance' not in config:
        config['building_tolerance'] = 0

    if 'results_file_prefix' not in config:
        config['results_file_prefix'] = None

    if 'results_file_dir' not in config:
        config['results_file_dir'] = None

    if 'analyze_results' not in config:
        config['analyze_results'] = None
    elif not isinstance(config['analyze_results'], (list, tuple)):
        config['analyze_results'] = [config['analyze_results']]

    if (config['simulation_mode'] == 'parallel') and ('processes' not in config):
        config['processes'] = None

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
        if 'skip_sumo' not in config['sumo']:
            config['sumo']['skip_sumo'] = False
        if 'directory' not in config['sumo']:
            config['sumo']['directory'] = 'sumo_data/'
        if 'veh_rate_factor' not in config['sumo']:
            config['sumo']['veh_rate_factor'] = None
        if 'coordinate_tls' not in config['sumo']:
            config['sumo']['coordinate_tls'] = True

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


def merge(orig, update, path=None):
    """Deep merges update into orig. (dict.update only shallow merges)"""

    if path is None: path = []
    for key in update:
        if key in orig:
            if isinstance(orig[key], dict) and isinstance(update[key], dict):
                merge(orig[key], update[key], path + [str(key)])
            elif orig[key] == update[key]:
                pass
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            orig[key] = update[key]
    return orig
