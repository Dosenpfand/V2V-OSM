"""Interface to SUMO – Simulation of Urban MObility, sumo.dlr.de"""

import sys
import os
import subprocess as sproc
import xml.etree.cElementTree as ET
import logging
import numpy as np
import shapely.geometry as geom
import utils
import osmnx as ox
import osmnx_addons as ox_a
import osm_xml
import vehicles


def simple_wrapper(place,
                   which_result=1,
                   max_count_veh=None,
                   duration=3600,
                   tls_settings=None,
                   directory='',
                   skip_if_exists=True,
                   veh_class='passenger'):
    """Generates and downloads all necessary files, runs a generic SUMO simulation
    and returns the vehicle traces"""

    filename_place = utils.string_to_filename(place)
    path_network_sumo = os.path.join(directory, filename_place + '.net.xml')
    filename_network_osm = filename_place + '_city.osm.xml'
    path_network_osm = os.path.join(
        directory, filename_network_osm)
    path_trips = os.path.join(
        directory, filename_place + '.' + veh_class + '.trips.xml')
    path_cfg = os.path.join(directory, filename_place + '.sumocfg')
    path_traces = os.path.join(directory, filename_place + '.traces.xml')

    if not (skip_if_exists and os.path.isfile(path_network_osm)):
        logging.info('Downloading street network from OpenStreetMap')

        if which_result is None:
            which_result = ox_a.which_result_polygon(place)

        download_streets_from_name(
            place, which_result=which_result, prefix=filename_place, directory=directory)

    else:
        logging.info('Skipping street network download from OpenStreetMap')

    if not (skip_if_exists and os.path.isfile(path_network_sumo)):
        logging.info('Generating SUMO street network')
        build_network(filename_network_osm, veh_class=veh_class,
                      prefix=filename_place, tls_settings=tls_settings, directory=directory)
    else:
        logging.info('Skipping SUMO street network generation')

    if not (skip_if_exists and os.path.isfile(path_trips)):
        logging.info('Generating trips')
        create_random_trips(place, seconds_end=duration, directory=directory)
    else:
        logging.info('Skipping trip generation')

    if not (skip_if_exists and os.path.isfile(path_cfg)):
        logging.info('Generating SUMO simulation configuration')
        gen_simulation_conf(
            place, directory=directory, seconds_end=duration, max_count_veh=max_count_veh)
    else:
        logging.info('Skipping SUMO simulation configuration generation')

    if not (skip_if_exists and os.path.isfile(path_traces)):
        logging.info('Running SUMO simulation')
        run_simulation(place, directory=directory)
    else:
        logging.info('Skipping SUMO simulation run')

    logging.info('Loading and parsing vehicle traces')
    traces = load_veh_traces(place, directory=directory)

    return traces


def gen_simulation_conf(place,
                        directory='',
                        seconds_end=None,
                        veh_class='passenger',
                        max_count_veh=None,
                        debug=False,
                        bin_dir=''):
    """Generates a SUMO simulation configuration file"""

    filename_place = utils.string_to_filename(place)
    path_cfg = os.path.join(directory, filename_place + '.sumocfg')
    path_network = filename_place + '.net.xml'
    path_trips = filename_place + '.' + veh_class + '.trips.xml'
    path_bin = os.path.join(bin_dir, 'sumo')

    arguments = [path_bin,
                 '-n', path_network,
                 '--duration-log.statistics',
                 '--device.rerouting.adaptation-steps', '180',
                 '--no-step-log',
                 '--save-configuration', path_cfg,
                 '--ignore-route-errors',
                 '-r', path_trips]

    if max_count_veh is not None:
        arguments += ['--max-num-vehicles', str(max_count_veh)]

    if seconds_end is not None:
        arguments += ['--end', str(seconds_end)]

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
                        intermediate_points=None,
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

    if intermediate_points is not None:
        arguments += ['--intermediate', str(intermediate_points)]

    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir,
                       stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def build_network(filename,
                  veh_class='passenger',
                  prefix=None,
                  tls_settings=None,
                  directory='',
                  debug=False,
                  script_dir='/usr/lib/sumo/tools'):
    """Converts a OpenStreetMap files to a SUMO street network file"""

    filepath = os.path.join(directory, filename)
    arguments = [script_dir + '/osmBuild.py', '-f', filepath, '-c', veh_class]

    if prefix is not None:
        arguments += ['-p', prefix]

    if directory != '':
        arguments += ['-d', directory]

    if isinstance(tls_settings, dict):
        # Taken from osmBuild.py
        netconvert_opts = '--geometry.remove,' + \
            '--roundabouts.guess,' + \
            '--ramps.guess,' + \
            '-v,' + \
            '--junctions.join,' + \
            '--tls.guess-signals,' + \
            '--tls.discard-simple,' + \
            '--tls.join,' + \
            '--output.original-names,' + \
            '--junctions.corner-detail,5,' + \
            '--output.street-names'

        if 'cycle_time' in tls_settings:
            netconvert_opts += ',--tls.cycle.time,' + \
                str(round(tls_settings['cycle_time']))

        # if 'green_time' in tls_settings:
        #     netconvert_opts += ',--tls.green.time,' + \
        #         str(round(tls_settings['green_time']))

        if 'yellow_time' in tls_settings:
            netconvert_opts += ',--tls.yellow.time,' + \
                str(round(tls_settings['yellow_time']))

        arguments += ['--netconvert-options', netconvert_opts]

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
                             directory='',
                             debug=False,
                             script_dir='/usr/lib/sumo/tools'):
    """Downloads a street data defined by it's id from OpennStreetMap
    with the SUMO helper script"""

    arguments = [script_dir + '/osmGet.py', '-a', str(area_id)]
    if prefix is not None:
        arguments += ['-p', prefix]
    if directory != '':
        arguments += ['-d', directory]
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
                               which_result=1,
                               prefix=None,
                               directory='',
                               debug=False,
                               use_sumo_downloader=False,
                               script_dir='/usr/lib/sumo/tools'):
    """Downloads a street data defined by it's name from OpennStreetMap"""

    if use_sumo_downloader:
        api_resp = ox.osm_polygon_download(
            place, limit=which_result, polygon_geojson=0)
        if not api_resp:
            raise RuntimeError('Place not found')
        area_id = api_resp[which_result - 1]['osm_id']
        exit_code = download_streets_from_id(
            area_id, prefix, directory, debug, script_dir)
        return exit_code

    else:
        if prefix is None:
            prefix = 'osm'

        file_name = prefix + '_city.osm.xml'
        file_path = os.path.join(directory, file_name)

        gdf_place = ox.gdf_from_place(place, which_result=which_result)
        polygon = gdf_place['geometry'].unary_union
        response = osm_xml.osm_net_download(polygon, network_type='drive')

        with open(file_path, 'wb') as file:
            return_code = file.write(response[0])
        return return_code


def load_veh_traces(place, directory=''):
    """Load parsed traces if they are available otherwise parse, return and save them"""

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
        traces_snapshot = np.zeros(
            len(timestep),
            dtype=[('time', 'float'),
                   ('id', 'uint'),
                   ('x', 'float'),
                   ('y', 'float')])
        for idx_veh_node, veh_node in enumerate(timestep):
            veh = veh_node.attrib
            veh_id = int(veh['id'][3:])
            traces_snapshot[idx_veh_node]['time'] = timestep.attrib['time']
            traces_snapshot[idx_veh_node]['id'] = veh_id
            traces_snapshot[idx_veh_node]['x'] = float(veh['x'])
            traces_snapshot[idx_veh_node]['y'] = float(veh['y'])
        traces_snapshot['x'] -= offsets[0]
        traces_snapshot['y'] -= offsets[1]
        traces[idx_timestep] = traces_snapshot
    return traces


def vehicles_from_traces(graph_streets, traces, time_idx):
    """ Builds a vehicles objects from the street graph, the SUMO traces and a time index"""

    snapshot = traces[time_idx]
    count_veh = snapshot.size
    points_vehs = np.zeros(count_veh, dtype=object)

    for veh_idx, vehicle in enumerate(snapshot):
        points_vehs[veh_idx] = geom.Point(vehicle['x'], vehicle['y'])

    vehs = vehicles.generate_vehs(
        graph_streets, street_idxs=None, points_vehs_in=points_vehs)

    return vehs


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
