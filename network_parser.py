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
import osmnx_addons as ox_a


def network_from_conf(in_key="default", config_file="network_definition.json"):
    """Load a network from the settings in a json file.

    Abstracts away load_network call.
    """
    conf = params_from_conf(in_key, config_file)
    return ox_a.load_network(conf["place"], conf["which_result"])


def params_from_conf(in_key="global_config",
                     config_file="network_definition.json"):
    """Load a parameter set from the given config_file.

    global_config: configuration params that are independent on chosen network.
    otherwise: network paramters for load_network and range and stuff."""
    with open(config_file, "r") as file_pointer:
        conf = json.load(file_pointer)
    return conf[in_key]
