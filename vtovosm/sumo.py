"""Interface to SUMO – Simulation of Urban MObility, sumo.dlr.de"""

import logging
import os
import subprocess as sproc
import sys
import xml.etree.cElementTree as ET

import numpy as np
import osmnx as ox
import shapely.geometry as geom

from . import osm_xml
from . import osmnx_addons as ox_a
from . import utils
from . import vehicles


def simple_wrapper(place,
                   which_result=1,
                   count_veh=None,
                   duration=3600,
                   warmup_duration=0,
                   max_speed=None,
                   tls_settings=None,
                   fringe_factor=None,
                   intermediate_points=None,
                   start_veh_simult=True,
                   coordinate_tls=True,
                   directory='sumo_data/',
                   skip_if_exists=True,
                   veh_class='passenger',
                   veh_rate_factor=None):
    """Generates and downloads all necessary files, runs a generic SUMO simulation
    and returns the vehicle traces."""

    filename_place = utils.string_to_filename(place)
    if count_veh is not None:
        filename_place_count = filename_place + '.' + str(count_veh)
    else:
        filename_place_count = filename_place
    path_network_sumo = os.path.join(directory, filename_place + '.net.xml')
    filename_network_osm = filename_place + '_city.osm.xml'
    path_network_osm = os.path.join(
        directory, filename_network_osm)
    path_trips = os.path.join(
        directory, filename_place_count + '.' + veh_class + '.trips.xml')
    path_tls = os.path.join(
        directory, filename_place + '.' + veh_class + '.tls.xml')
    path_cfg = os.path.join(directory, filename_place_count + '.sumocfg')
    path_traces = os.path.join(directory, filename_place_count + '.traces.xml')

    # Create the output directory if it does not exist
    if not os.path.isdir(directory):
        os.makedirs(directory)

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
        if count_veh is not None:
            # Generate more trips than needed because validation will throw some away
            if veh_rate_factor is None:
                veh_rate_factor = 0.5
            veh_rate = duration / count_veh * veh_rate_factor
        else:
            veh_rate = 1

        create_random_trips(place,
                            directory=directory,
                            file_suffix=str(count_veh),
                            fringe_factor=fringe_factor,
                            veh_period=veh_rate,
                            intermediate_points=intermediate_points)
        modify_trips(place,
                     directory=directory,
                     file_suffix=str(count_veh),
                     start_all_at_zero=start_veh_simult,
                     rename_ids=True,
                     limit_veh_count=count_veh,
                     max_speed=max_speed)
    else:
        logging.info('Skipping trip generation')

    if coordinate_tls and not (skip_if_exists and os.path.isfile(path_tls)):
        logging.info('Generating SUMO TLS coordination')
        if count_veh is not None:
            count_veh_tls = int(np.ceil(count_veh / 10))
        else:
            count_veh_tls = None

        generate_tls_coordination(place,
                                  directory=directory,
                                  file_suffix=str(count_veh),
                                  count_veh=count_veh_tls)
    else:
        logging.info('Skipping SUMO TLS coordination')

    if not (skip_if_exists and os.path.isfile(path_cfg)):
        logging.info('Generating SUMO simulation configuration')
        gen_simulation_conf(
            place,
            directory=directory,
            file_suffix=str(count_veh),
            seconds_end=duration,
            max_count_veh=count_veh,
            coordinate_tls=coordinate_tls)
    else:
        logging.info('Skipping SUMO simulation configuration generation')

    if not (skip_if_exists and os.path.isfile(path_traces)):
        logging.info('Running SUMO simulation')
        run_simulation(place, file_suffix=str(count_veh), directory=directory)
    else:
        logging.info('Skipping SUMO simulation run')

    logging.info('Loading parsing and cleaning vehicle traces')
    traces = load_veh_traces(place,
                             file_suffix=str(count_veh),
                             directory=directory,
                             delete_first_n=warmup_duration,
                             count_veh=count_veh)

    return traces


def gen_simulation_conf(place,
                        directory='',
                        file_suffix=None,
                        seconds_end=None,
                        veh_class='passenger',
                        max_count_veh=None,
                        coordinate_tls=True,
                        use_route_file=True,
                        debug=False,
                        bin_dir=''):
    """Generates a SUMO simulation configuration file"""

    filename_place = utils.string_to_filename(place)

    if file_suffix is None:
        filename_place_suffix = filename_place
    else:
        filename_place_suffix = filename_place + '.' + str(file_suffix)

    path_cfg = os.path.join(directory, filename_place_suffix + '.sumocfg')
    path_bin = os.path.join(bin_dir, 'sumo')
    filename_network = filename_place + '.net.xml'
    filename_trips = filename_place_suffix + '.' + veh_class + '.trips.xml'
    filename_tls = filename_place + '.' + veh_class + '.tls.xml'
    filename_routes = filename_place_suffix + '.' + veh_class + '.rou.xml'

    arguments = [path_bin,
                 '-n', filename_network,
                 '--duration-log.statistics',
                 '--device.rerouting.adaptation-steps', '180',
                 '--no-step-log',
                 '--save-configuration', path_cfg,
                 '--ignore-route-errors']

    if max_count_veh is not None:
        arguments += ['--max-num-vehicles', str(max_count_veh)]

    if seconds_end is not None:
        arguments += ['--end', str(seconds_end)]

    if coordinate_tls:
        arguments += ['-a', filename_tls]

    if use_route_file:
        arguments += ['-r', filename_routes]
    else:
        arguments += ['-r', filename_trips]

    proc = sproc.Popen(arguments, stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        utils.print_nnl(err_text.decode(), file=sys.stderr)
        raise RuntimeError('SUMO quit with nonzero exit code')

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def run_simulation(place, directory='', file_suffix=None, debug=False, bin_dir=''):
    """Runs a SUMO simulations and saves the vehicle traces"""

    filename_place = utils.string_to_filename(place)

    if file_suffix is None:
        filename_place_suffix = filename_place
    else:
        filename_place_suffix = filename_place + '.' + str(file_suffix)

    path_cfg = os.path.join(directory, filename_place_suffix + '.sumocfg')
    path_traces = os.path.join(
        directory, filename_place_suffix + '.traces.xml')
    path_bin = os.path.join(bin_dir, 'sumo')

    arguments = [path_bin,
                 '-c', path_cfg,
                 '--fcd-output', path_traces]

    proc = sproc.Popen(arguments, stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        utils.print_nnl(err_text.decode(), file=sys.stderr)
        raise RuntimeError('SUMO quit with nonzero exit code')

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def modify_trips(place,
                 directory='',
                 file_suffix=None,
                 start_all_at_zero=False,
                 rename_ids=False,
                 limit_veh_count=None,
                 max_speed=None,
                 modify_routes=True,
                 veh_class='passenger',
                 prefix='veh'):
    """Modifies the randomly generated trips according to the parameters"""

    filename_place = utils.string_to_filename(place)

    if file_suffix is None:
        filename_place_suffix = filename_place
    else:
        filename_place_suffix = filename_place + '.' + str(file_suffix)

    path_trips = os.path.join(
        directory, filename_place_suffix + '.' + veh_class + '.trips.xml')
    path_routes = os.path.join(
        directory, filename_place_suffix + '.' + veh_class + '.rou.xml')

    # Modify trips file
    tree = ET.parse(path_trips)
    root = tree.getroot()

    if limit_veh_count is not None:
        for trip in root.findall('trip')[limit_veh_count:]:
            root.remove(trip)

    if start_all_at_zero:
        for trip in root.findall('trip'):
            trip.attrib['depart'] = '0.00'

    if rename_ids:
        for idx, trip in enumerate(root.findall('trip')):
            trip.attrib['id'] = prefix + str(idx)

    if max_speed is not None:
        for vtype in root.findall('vType'):
            vtype.attrib['maxSpeed'] = str(max_speed)

    tree.write(path_trips, 'UTF-8')

    # Modify routes file
    if not modify_routes:
        return

    tree = ET.parse(path_routes)
    root = tree.getroot()

    if limit_veh_count is not None:
        for trip in root.findall('vehicle')[limit_veh_count:]:
            root.remove(trip)

    if start_all_at_zero:
        for trip in root.findall('vehicle'):
            trip.attrib['depart'] = '0.00'

    if rename_ids:
        for idx, trip in enumerate(root.findall('vehicle')):
            trip.attrib['id'] = prefix + str(idx)

    if max_speed is not None:
        for vtype in root.findall('vType'):
            vtype.attrib['maxSpeed'] = str(max_speed)

    tree.write(path_routes, 'UTF-8')


def create_random_trips(place,
                        directory='',
                        file_suffix=None,
                        random_seed=42,
                        seconds_end=3600,
                        fringe_factor=None,
                        veh_period=1,
                        veh_class='passenger',
                        prefix='veh',
                        min_dist=300,
                        intermediate_points=None,
                        debug=False,
                        script_dir=None):
    """Creates random vehicle trips on a street network"""

    filename_place = utils.string_to_filename(place)

    if file_suffix is None:
        filename_place_suffix = filename_place
    else:
        filename_place_suffix = filename_place + '.' + str(file_suffix)

    path_network = os.path.join(
        directory, filename_place + '.net.xml')
    path_routes = os.path.join(
        directory, filename_place_suffix + '.' + veh_class + '.rou.xml')
    path_trips = os.path.join(
        directory, filename_place_suffix + '.' + veh_class + '.trips.xml')

    if script_dir is None:
        script_dir = search_tool_dir()

    arguments = [os.path.join(script_dir, 'randomTrips.py'),
                 '-n', path_network,
                 '-s', str(random_seed),
                 '-e', str(seconds_end),
                 '-p', str(veh_period),
                 '-r', path_routes,
                 '-o', path_trips,
                 '--vehicle-class', veh_class,
                 '--vclass', veh_class,
                 '--prefix', prefix,
                 '--min-distance', str(min_dist),
                 '--validate']

    if intermediate_points is not None:
        arguments += ['--intermediate', str(intermediate_points)]

    if fringe_factor is not None:
        arguments += ['--fringe-factor', str(fringe_factor)]

    proc = sproc.Popen(arguments, stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        utils.print_nnl(err_text.decode(), file=sys.stderr)
        raise RuntimeError('Trip generation script quit with nonzero exit code')

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
                  script_dir=None,
                  remove_isolated=True):
    """Converts a OpenStreetMap files to a SUMO street network file"""

    filepath = os.path.join(directory, filename)

    if script_dir is None:
        script_dir = search_tool_dir()

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

        if remove_isolated:
            netconvert_opts += ',--remove-edges.isolated'

        if ('cycle_time' in tls_settings) and ('green_time' in tls_settings):
            raise RuntimeError(
                'Cycle time and green time can not be set simultaneosly')

        if 'cycle_time' in tls_settings:
            netconvert_opts += ',--tls.cycle.time,' + \
                               str(round(tls_settings['cycle_time']))

        if 'green_time' in tls_settings:
            netconvert_opts += ',--tls.green.time,' + \
                               str(round(tls_settings['green_time']))

        if 'yellow_time' in tls_settings:
            netconvert_opts += ',--tls.yellow.time,' + \
                               str(round(tls_settings['yellow_time']))

        arguments += ['--netconvert-options', netconvert_opts]

    proc = sproc.Popen(arguments, stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        utils.print_nnl(err_text.decode(), file=sys.stderr)
        raise RuntimeError('Network build script quit with nonzero exit code')

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def generate_tls_coordination(place,
                              directory='',
                              file_suffix=None,
                              veh_class='passenger',
                              count_veh=None,
                              debug=False,
                              script_dir=None):
    """Generates a traffic light system coordination"""

    filename_place = utils.string_to_filename(place)

    if file_suffix is None:
        filename_place_suffix = filename_place
    else:
        filename_place_suffix = filename_place + '.' + str(file_suffix)

    path_network = os.path.join(
        directory, filename_place + '.net.xml')
    path_tls = os.path.join(
        directory, filename_place + '.' + veh_class + '.tls.xml')

    if count_veh is None:
        path_routes = os.path.join(
            directory, filename_place_suffix + '.' + veh_class + '.rou.xml')
    else:
        path_routes_full = os.path.join(
            directory, filename_place_suffix + '.' + veh_class + '.rou.xml')
        path_routes = os.path.join(
            directory, filename_place_suffix + '.' + veh_class + '.rou_part.xml')
        tree = ET.parse(path_routes_full)
        root = tree.getroot()
        for vehicle in root.findall('vehicle')[count_veh:]:
            root.remove(vehicle)

        tree.write(path_routes, 'UTF-8')

    if script_dir is None:
        script_dir = search_tool_dir()

    arguments = [script_dir + '/tlsCoordinator.py',
                 '-n', path_network,
                 '-r', path_routes,
                 '-o', path_tls]

    proc = sproc.Popen(arguments, stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        utils.print_nnl(err_text.decode(), file=sys.stderr)
        raise RuntimeError('TLS coordination script quit with nonzero exit code')

    if debug:
        utils.print_nnl(out_text.decode())
    utils.print_nnl(err_text.decode(), file=sys.stderr)

    return exit_code


def download_streets_from_id(area_id,
                             prefix=None,
                             directory='',
                             debug=False,
                             script_dir=None):
    """Downloads a street data defined by it's id from OpenStreetMap
    with the SUMO helper script"""

    if script_dir is None:
        script_dir = search_tool_dir()

    arguments = [script_dir + '/osmGet.py', '-a', str(area_id)]
    if prefix is not None:
        arguments += ['-p', prefix]
    if directory != '':
        arguments += ['-d', directory]

    proc = sproc.Popen(arguments, stdout=sproc.PIPE, stderr=sproc.PIPE)
    out_text, err_text = proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        utils.print_nnl(err_text.decode(), file=sys.stderr)
        raise RuntimeError('OSM download script quit with nonzero exit code')

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
                               script_dir=None):
    """Downloads a street data defined by it's name from OpenStreetMap"""

    # Setup OSMnx
    ox_a.setup()

    if use_sumo_downloader:
        if script_dir is None:
            script_dir = search_tool_dir()

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


def load_veh_traces(place, directory='', file_suffix=None, delete_first_n=0, count_veh=None):
    """Load parsed traces if they are available otherwise parse,
    clean up (if requested) and save them. Return the traces"""

    filename_place = utils.string_to_filename(place)

    if file_suffix is None:
        filename_place_suffix = filename_place
    else:
        filename_place_suffix = filename_place + '.' + str(file_suffix)

    path_and_prefix = os.path.join(directory, filename_place)
    path_and_prefix_suffix = os.path.join(directory, filename_place_suffix)

    filename_traces_npy = path_and_prefix_suffix + '.traces.pickle.xz'
    filename_traces_xml = path_and_prefix_suffix + '.traces.xml'
    filename_network = path_and_prefix + '.net.xml'

    if os.path.isfile(filename_traces_npy):
        traces = utils.load(filename_traces_npy)
    else:
        coord_offsets = get_coordinates_offset(filename_network)
        traces = parse_veh_traces(filename_traces_xml, coord_offsets)
        traces = clean_veh_traces(
            traces, delete_first_n=delete_first_n, count_veh=count_veh)
        utils.save(traces, filename_traces_npy)
    return traces


def clean_veh_traces(veh_traces, delete_first_n=0, count_veh=None):
    """Cleans up vehicle traces according to the given parameters"""

    # delete first n snapshots
    veh_traces = veh_traces[delete_first_n:]

    # Delete snapshots with wrong number of vehicles
    if count_veh is not None:
        retain_mask = np.ones(veh_traces.size, dtype=bool)
        for idx, snapshot in enumerate(veh_traces):
            if snapshot.size != count_veh:
                retain_mask[idx] = False
                logging.warning(
                    'Vehicle traces snapshot {:d} has wrong size, discarding'.format(idx))
        veh_traces = veh_traces[retain_mask]

    return veh_traces


def parse_veh_traces(filename, offsets=(0, 0), sort=True):
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

        if sort:
            traces_snapshot.sort(order='id')

        traces[idx_timestep] = traces_snapshot

    return traces


def vehicles_from_traces(graph_streets, snapshot):
    """ Builds a vehicles objects from the street graph
    and a snapshot of the SUMO vehicle traces"""

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


def search_tool_dir():
    """Searches for the SUMO tools directory"""

    paths = ['sumo/sumo/tools',         # Local installation
             '/usr/lib/sumo/tools',     # Arch Linux default location
             '/usr/share/sumo/tools']   # Debian default location
    for path in paths:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError('Could not find the SUMO tools directory')
