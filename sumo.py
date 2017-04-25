"""Interface to SUMO – Simulation of Urban MObility, sumo.dlr.de"""

import os
import subprocess as sproc
import xml.etree.cElementTree as ET
import numpy as np
import utils
import osmnx as ox


def download_and_build_network(place,
                               prefix=None,
                               out_dir=None,
                               veh_class='passenger',
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

    download_streets_from_name(place, file_prefix, out_dir, script_path)
    build_network(out_dir + file_prefix + '_city.osm.xml',
                  veh_class, file_prefix, out_dir, script_path)


def build_network(filename,
                  veh_class='passenger',
                  prefix=None,
                  out_dir=None,
                  script_dir='/usr/lib/sumo/tools'):
    """Converts a OpenStreetMap files to a SUMO street network file"""

    arguments = [script_dir + '/osmBuild.py', '-f', filename, '-c', veh_class]
    if prefix is not None:
        arguments += ['-p', prefix]
    if out_dir is not None:
        arguments += ['-d', out_dir]
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir)
    exit_code = proc.wait()
    return exit_code


def download_streets_from_id(area_id,
                             prefix=None,
                             out_dir=None,
                             script_dir='/usr/lib/sumo/tools'):
    """Downloads a street data defined by it's id from OpennStreetMap
    with the SUMO helper script"""

    arguments = [script_dir + '/osmGet.py', '-a', str(area_id)]
    if prefix is not None:
        arguments += ['-p', prefix]
    if out_dir is not None:
        arguments += ['-d', out_dir]
    working_dir = os.path.dirname(os.path.abspath(__file__))

    proc = sproc.Popen(arguments, cwd=working_dir)
    exit_code = proc.wait()
    return exit_code


def download_streets_from_name(place,
                               prefix=None,
                               out_dir=None,
                               script_path='/usr/lib/sumo/tools/osmGet.py'):
    """Downloads a street data defined by it's name from OpennStreetMap
    with the SUMO helper script"""

    api_resp = ox.osm_polygon_download(place, polygon_geojson=0)
    area_id = api_resp[0]['osm_id']
    exit_code = download_streets_from_id(area_id, prefix, out_dir, script_path)
    return exit_code


def load_veh_traces(place):
    """Load parsed traces if they are available otherwise parse, return and save them"""

    file_prefix = 'sumo_traces/{}'.format(utils.string_to_filename(place))
    filename_traces_npy = file_prefix + '.traces.npy'
    filename_traces_xml = file_prefix + '.traces.xml'
    filename_network = file_prefix + '.net.xml'

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
