"""Interface to SUMO – Simulation of Urban MObility, sumo.dlr.de"""

import sys
import os
import subprocess as sproc
import xml.etree.cElementTree as ET
import numpy as np
import utils
import osmnx as ox


def sumo_simple_simulation_wrapper(place, directory=''):
    """Generates and downloads all necessary files, runs a generic SUMO simulation
    and returns the vehicles traces"""


def gen_simulation_conf(place, directory='', veh_class='passenger', debug=False, bin_dir=''):
    """Generates a SUMO simulation configuration file"""

    filename_place = utils.string_to_filename(place)
    path_network = os.path.join(directory, filename_place + '.net.xml')
    path_cfg = os.path.join(directory, filename_place + '.sumocfg')
    path_trips = os.path.join(
        directory, filename_place + '.' + veh_class + '.trips.xml')
    path_bin = os.path.join(bin_dir, 'sumo')

    arguments = [path_bin,
                 '-n', path_network,
                 '--duration-log.statistics',
                 '--device.rerouting.adaptation-steps', '180',
                 '--no-step-log',
                 '--save-configuration', path_cfg,
                 '--ignore-route-errors',
                 '-r', path_trips]
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir,
                       stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def run_simulation(place, directory='', debug=False, bin_dir=''):
    """Runs a SUMO simulations and saves the vehicle traces"""

    filename_place = utils.string_to_filename(place)
    path_cfg = os.path.join(directory, filename_place + '.sumocfg')
    path_traces = os.path.join(directory, filename_place + '.traces.xml')
    path_bin = os.path.join(bin_dir, 'sumo')

    arguments = [path_bin,
                 '-c', path_cfg,
                 '--fcd-output', path_traces]
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir,
                       stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def create_random_trips(place,
                        directory='',
                        random_seed=42,
                        seconds_end=3600,
                        fringe_factor=5,
                        veh_period=1,
                        veh_class='passenger',
                        prefix='veh',
                        min_dist=300,
                        debug=False,
                        script_dir='/usr/lib/sumo/tools'):
    """Creates random vehicle trips on a street network"""

    filename_place = utils.string_to_filename(place)
    path_network = os.path.join(
        directory, filename_place + '.net.xml')
    path_routes = os.path.join(
        directory, filename_place + '.' + veh_class + '.rou.xml')
    path_trips = os.path.join(
        directory, filename_place + '.' + veh_class + '.trips.xml')

    arguments = [os.path.join(script_dir, 'randomTrips.py'),
                 '-n', path_network,
                 '-s', str(random_seed),
                 '-e', str(seconds_end),
                 '-p', str(veh_period),
                 '--fringe-factor', str(fringe_factor),
                 '-r', path_routes,
                 '-o', path_trips,
                 '--vehicle-class', veh_class,
                 '--vclass', veh_class,
                 '--prefix', prefix,
                 '--min-distance', str(min_dist),
                 '--validate']
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir,
                       stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def download_and_build_network(place,
                               prefix=None,
                               out_dir=None,
                               veh_class='passenger',
                               debug=False,
                               script_path='/usr/lib/sumo/tools'):
    """Donloads the street data from OpenStreetMap and builds a SUMO street network file"""

    if prefix is None:
        file_prefix = utils.string_to_filename(place)
    else:
        file_prefix = prefix

    if out_dir is None:
        out_dir = './'
    else:
        out_dir = out_dir + '/'

    download_streets_from_name(place, file_prefix, out_dir, debug, script_path)
    build_network(out_dir + file_prefix + '_city.osm.xml',
                  veh_class, file_prefix, out_dir, debug, script_path)


def build_network(filename,
                  veh_class='passenger',
                  prefix=None,
                  out_dir=None,
                  debug=False,
                  script_dir='/usr/lib/sumo/tools'):
    """Converts a OpenStreetMap files to a SUMO street network file"""

    arguments = [script_dir + '/osmBuild.py', '-f', filename, '-c', veh_class]
    if prefix is not None:
        arguments += ['-p', prefix]
    if out_dir is not None:
        arguments += ['-d', out_dir]
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir,
                       stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def download_streets_from_id(area_id,
                             prefix=None,
                             out_dir=None,
                             debug=False,
                             script_dir='/usr/lib/sumo/tools'):
    """Downloads a street data defined by it's id from OpennStreetMap
    with the SUMO helper script"""

    arguments = [script_dir + '/osmGet.py', '-a', str(area_id)]
    if prefix is not None:
        arguments += ['-p', prefix]
    if out_dir is not None:
        arguments += ['-d', out_dir]
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir,
                       stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def download_streets_from_name(place,
                               prefix=None,
                               out_dir=None,
                               debug=False,
                               script_dir='/usr/lib/sumo/tools'):
    """Downloads a street data defined by it's name from OpennStreetMap
    with the SUMO helper script"""

    # TODO: does not always work. e.g. 'Upper Westside - New York - USA'. Use
    # other api? check osmnx!
    api_resp = ox.osm_polygon_download(place, polygon_geojson=0)
    if len(api_resp) == 0:
        raise RuntimeError('Place not found')
    area_id = api_resp[0]['osm_id']
    exit_code = download_streets_from_id(
        area_id, prefix, out_dir, debug, script_dir)
    return exit_code


def load_veh_traces(place, directory=''):
    """Load parsed traces if they are available otherwise parse, return and save them"""

    # TODO: test!
    path_and_prefix = os.path.join(directory, utils.string_to_filename(place))
    filename_traces_npy = path_and_prefix + '.traces.npy'
    filename_traces_xml = path_and_prefix + '.traces.xml'
    filename_network = path_and_prefix + '.net.xml'

    if os.path.isfile(filename_traces_npy):
        traces = np.load(filename_traces_npy)
    else:
        coord_offsets = get_coordinates_offset(filename_network)
        traces = parse_veh_traces(filename_traces_xml, coord_offsets)
        np.save(filename_traces_npy, traces)
    return traces


def parse_veh_traces(filename, offsets=(0, 0)):
    """Parses a SUMO traces XML file and returns a numpy array"""

    tree = ET.parse(filename)
    root = tree.getroot()

    traces = np.zeros(len(root), dtype=object)
    for idx_timestep, timestep in enumerate(root):
        trace_timestep = np.zeros(
            len(timestep),
            dtype=[('time', 'float'),
                   ('id', 'uint'),
                   ('x', 'float'),
                   ('y', 'float')])
        for idx_veh_node, veh_node in enumerate(timestep):
            veh = veh_node.attrib
            veh_id = int(veh['id'][3:])
            trace_timestep[idx_veh_node]['time'] = timestep.attrib['time']
            trace_timestep[idx_veh_node]['id'] = veh_id
            trace_timestep[idx_veh_node]['x'] = float(veh['x'])
            trace_timestep[idx_veh_node]['y'] = float(veh['y'])
        trace_timestep['x'] -= offsets[0]
        trace_timestep['y'] -= offsets[1]
        traces[idx_timestep] = trace_timestep
    return traces


def get_coordinates_offset(filename):
    """Retrieves the x and y offset of the UTM projection from the SUMO net file"""

    tree = ET.parse(filename)
    root = tree.getroot()
    location = root.find('location')
    offset_string = location.attrib['netOffset']
    offset_string_x, offset_string_y = offset_string.split(',')
    offset_x = float(offset_string_x)
    offset_y = float(offset_string_y)
    offsets = [offset_x, offset_y]
    return offsets


def min_max_coords(traces):
    """Determines the min and max x and y coordinates of all vehicle traces"""

    # TODO: better performance?
    x_min, x_max = traces[0][0]['x'], traces[0][0]['x']
    y_min, y_max = traces[0][0]['y'], traces[0][0]['y']
    for trace in traces:
        if np.size(trace) == 0:
            continue
        x_min_iter = trace['x'].min()
        x_max_iter = trace['x'].max()
        y_min_iter = trace['y'].min()
        y_max_iter = trace['y'].max()
        if x_min_iter < x_min:
            x_min = x_min_iter
        if x_max_iter > x_max:
            x_max = x_max_iter
        if y_min_iter < y_min:
            y_min = y_min_iter
        if y_max_iter > y_max:
            y_max = y_max_iter

    return x_min, x_max, y_min, y_max
