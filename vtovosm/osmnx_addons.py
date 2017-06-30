""" Additional functions missing in the OSMnx package"""

import logging
import os

import geopandas as gpd
import numpy as np
import osmnx as ox
import shapely.geometry as geom
import shapely.ops as ops

from . import propagation as prop
from . import utils


def setup():
    """Sets up OSMnx"""

    # TODO: still outputs all and not only >= loglevel!
    logger = logging.getLogger()
    ox.config(log_console=False, log_file=os.devnull, log_name=logger.name, use_cache=True)


def load_network(place, which_result=1, overwrite=False, tolerance=0):
    """Generates streets and buildings"""

    # Generate filenames
    file_prefix = 'data/{}'.format(utils.string_to_filename(place))
    filename_data_streets = 'data/{}_streets.pickle.xz'.format(
        utils.string_to_filename(place))
    filename_data_boundary = 'data/{}_boundary.pickle.xz'.format(
        utils.string_to_filename(place))
    filename_data_wave = 'data/{}_wave.pickle.xz'.format(
        utils.string_to_filename(place))
    filename_data_buildings = 'data/{}_buildings.pickle.xz'.format(
        utils.string_to_filename(place))

    # Create the output directory if it does not exist
    if not os.path.isdir('data/'):
        os.makedirs('data/')

    if not overwrite and \
            os.path.isfile(filename_data_streets) and \
            os.path.isfile(filename_data_buildings) and \
            os.path.isfile(filename_data_boundary):
        # Load from file
        time_start = utils.debug(None, 'Loading data from disk')
        data = load_place(file_prefix, tolerance=tolerance)
    else:
        # Load from internet
        time_start = utils.debug(None, 'Loading data from the internet')
        data = download_place(place, which_result=which_result, tolerance=tolerance)

    graph_streets = data['streets']
    gdf_buildings = data['buildings']
    gdf_boundary = data['boundary']
    add_geometry(graph_streets)

    utils.debug(time_start)

    # Generate wave propagation graph:
    # Vehicles are placed in a undirected version of the graph because electromagnetic
    # waves do not respect driving directions
    if not overwrite and os.path.isfile(filename_data_wave):
        # Load from file
        time_start = utils.debug(None, 'Loading graph for wave propagation')
        graph_streets_wave = utils.load(filename_data_wave)
    else:
        # Generate
        time_start = utils.debug(None, 'Generating graph for wave propagation')
        graph_streets_wave = graph_streets.to_undirected()
        prop.add_edges_if_los(graph_streets_wave, gdf_buildings)
        utils.save(graph_streets_wave, filename_data_wave)

    utils.debug(time_start)

    network = {'graph_streets': graph_streets,
               'graph_streets_wave': graph_streets_wave,
               'gdf_buildings': gdf_buildings,
               'gdf_boundary': gdf_boundary}

    return network


def download_place(place, network_type='drive', file_prefix=None, which_result=1, project=True, tolerance=0):
    """ Downloads streets and buildings for a place, saves the data to disk and returns them """

    if file_prefix is None:
        file_prefix = 'data/{}'.format(utils.string_to_filename(place))

    if which_result is None:
        which_result = which_result_polygon(place)

    # Streets
    streets = ox.graph_from_place(
        place, network_type=network_type, which_result=which_result)
    if project:
        streets = ox.project_graph(streets)
    filename_streets = '{}_streets.pickle.xz'.format(file_prefix)
    utils.save(streets, filename_streets)

    # Boundary and buildings
    boundary = ox.gdf_from_place(place, which_result=which_result)
    polygon = boundary['geometry'].iloc[0]
    buildings = ox.create_buildings_gdf(polygon)
    if project:
        buildings = ox.project_gdf(buildings)
        boundary = ox.project_gdf(boundary)

    # Save buildings
    filename_buildings = '{}_buildings.pickle.xz'.format(file_prefix)
    utils.save(buildings, filename_buildings)

    # Build and save simplified buildings
    if tolerance != 0:
        filename_buildings_simpl = '{}_buildings_{:.2f}.pickle.xz'.format(file_prefix, tolerance)
        buildings = simplify_buildings(buildings)
        utils.save(buildings, filename_buildings_simpl)

    # Save boundary
    filename_boundary = '{}_boundary.pickle.xz'.format(file_prefix)
    utils.save(boundary, filename_boundary)

    # Return data
    data = {'streets': streets, 'buildings': buildings, 'boundary': boundary}
    return data


def load_place(file_prefix, tolerance=0):
    """ Loads previously downloaded street and building data of a place"""

    filename_buildings = '{}_buildings.pickle.xz'.format(file_prefix)

    if tolerance == 0:
        buildings = utils.load(filename_buildings)
    else:
        filename_buildings_simpl = '{}_buildings_{:.2f}.pickle.xz'.format(file_prefix, tolerance)
        if os.path.isfile(filename_buildings_simpl):
            buildings = utils.load(filename_buildings_simpl)
        else:
            buildings_compl = utils.load(filename_buildings)
            buildings = simplify_buildings(buildings_compl)
            utils.save(buildings, filename_buildings_simpl)

    filename_streets = '{}_streets.pickle.xz'.format(file_prefix)
    streets = utils.load(filename_streets)

    filename_boundary = '{}_boundary.pickle.xz'.format(file_prefix)
    boundary = utils.load(filename_boundary)

    place = {'streets': streets, 'buildings': buildings, 'boundary': boundary}
    return place


def add_geometry(streets):
    """ Adds geometry object to the edges of the graph where they are missing"""
    for u_node, v_node, data in streets.edges(data=True):
        if 'geometry' not in data:
            coord_x1 = streets.node[u_node]['x']
            coord_y1 = streets.node[u_node]['y']
            coord_x2 = streets.node[v_node]['x']
            coord_y2 = streets.node[v_node]['y']
            data['geometry'] = geom.LineString(
                [(coord_x1, coord_y1), (coord_x2, coord_y2)])


def check_geometry(streets):
    """ Checks if all edges of the graph have a geometry object"""

    complete = True
    for _, _, data in streets.edges(data=True):
        if 'geometry' not in data:
            complete = False
            break

    return complete


def which_result_polygon(query, limit=5):
    """Determines the first which_result value that returns a polygon from the nominatim API"""

    response = ox.osm_polygon_download(query, limit=limit, polygon_geojson=1)
    for index, result in enumerate(response):
        if result['geojson']['type'] == 'Polygon':
            return index + 1
    return None


def simplify_buildings(gdf_buildings, tolerance=1, merge_by_fill=True):
    """Simplifies the building polygons by reducing the number of edges.
    Notes: The resulting deviation can be larger than tolerance, because both merging and simplifying use tolerance."""

    geoms_list = gdf_buildings.geometry.tolist()
    geoms_list_comb = []

    # Merge polygons if they are near each other
    for idx1 in range(len(geoms_list)):
        geom1 = geoms_list[idx1]

        if geom1 is None:
            continue
        elif not isinstance(geom1, geom.Polygon):
            geoms_list_comb.append(geom1)
            continue

        # Because of previous merges we need to check from the beginning and not from idx+1
        for idx2 in range(len(geoms_list)):
            geom2 = geoms_list[idx2]

            if idx1 == idx2:
                continue
            elif geom2 is None:
                continue
            elif not isinstance(geom2, geom.Polygon):
                continue

            dist = geom1.distance(geom2)

            if dist > tolerance:
                continue

            if merge_by_fill:
                geom_union = merge_polygons_by_fill(geom1, geom2)
            else:
                geom_union = merge_polygons_by_buffer(geom1, geom2)

            # If the union is 2 separate polygons we keep them otherwise we save the union
            if not isinstance(geom_union, geom.MultiPolygon):
                geom1 = geom_union
                geoms_list[idx2] = None

        geoms_list[idx1] = geom1
        geoms_list_comb.append(geom1)

    # Remove interiors of polygons
    geoms_list_ext = remove_interior_polygons(geoms_list_comb)

    # Simplify polygons
    geoms_list_simpl = simplify_polygons(geoms_list_ext, tolerance=tolerance)

    # Build a new GDF
    buildings = {}
    for idx, geometry in enumerate(geoms_list_simpl):
        building = {'id': idx,
                    'geometry': geometry}
        buildings[idx] = building

    gdf_buildings_opt = gpd.GeoDataFrame(buildings).T

    return gdf_buildings_opt


def simplify_polygons(polygons_list, tolerance=1):
    """Simplifies a list of polygons"""

    polygons_list_simpl = []
    for geometry in polygons_list:
        if not isinstance(geometry, geom.Polygon):
            polygons_list_simpl.append(geometry)
            continue

        geometry_simpl = geometry.simplify(tolerance, preserve_topology=False)

        if isinstance(geometry_simpl, geom.MultiPolygon):
            for poly in geometry_simpl:
                if not poly.is_empty:
                    polygons_list_simpl.append(poly)
        else:
            if not geometry_simpl.is_empty:
                polygons_list_simpl.append(geometry_simpl)
            else:
                polygons_list_simpl.append(geometry)

    return polygons_list_simpl


def remove_interior_polygons(polygons_list):
    """Removes all interiors of a list of polygons"""

    polygons_list_exterior = []
    for geometry in polygons_list:
        if not isinstance(geometry, geom.Polygon):
            polygons_list_exterior.append(geometry)
        else:
            poly_simp = geom.Polygon(geometry.exterior)
            polygons_list_exterior.append(poly_simp)

    return polygons_list_exterior


def merge_polygons_by_fill(polygon1, polygon2):
    """Merges 2 polygons by searching for the 2 nearest nodes on each and constructing a square to fill the gap
    region"""

    if polygon1.intersects(polygon2):
        geom_union = ops.unary_union([polygon1, polygon2])
        return geom_union

    coords1 = np.array(polygon1.exterior.coords.xy)
    coords2 = np.array(polygon2.exterior.coords.xy)
    points1 = [geom.Point(coord) for coord in coords1.T][:-1]
    points2 = [geom.Point(coord) for coord in coords2.T][:-1]

    # Find pair of closest edges
    min_dist_1 = np.inf
    min_idx1_1 = None
    min_idx2_1 = None
    for idx1, point1 in enumerate(points1):
        for idx2, point2 in enumerate(points2):
            cur_dist = point1.distance(point2)
            if cur_dist < min_dist_1:
                min_dist_1 = cur_dist
                min_idx1_1 = idx1
                min_idx2_1 = idx2

    # Find pair of 2nd closest edges
    min_dist_2 = np.inf
    min_idx1_2 = None
    min_idx2_2 = None
    for idx1, point1 in enumerate(points1):
        if (idx1 == min_idx1_1) or point1.almost_equals(points1[min_idx1_1]):
            continue
        for idx2, point2 in enumerate(points2):
            if (idx2 == min_idx2_1) or point2.almost_equals(points2[min_idx2_1]):
                continue
            cur_dist = point1.distance(point2)
            if cur_dist < min_dist_2:
                min_dist_2 = cur_dist
                min_idx1_2 = idx1
                min_idx2_2 = idx2

    # Generate fill square
    points_fill = [points1[min_idx1_1], points2[min_idx2_1], points2[min_idx2_2], points1[min_idx1_2]]
    points_fill_idxs = [(0,1,2,3), (1,0,2,3), (0,2,1,3)]

    poly_fill = None
    for idxs in points_fill_idxs:
        points_fill_iter = [points_fill[idx] for idx in idxs]
        coords_fill = [point.coords[:][0] for point in points_fill_iter]
        poly_fill = geom.Polygon(coords_fill)
        if poly_fill.is_valid:
            break

    # Build union of 3 polygons
    geom_union = ops.unary_union([polygon1, poly_fill, polygon2]).simplify(0)
    return geom_union


def merge_polygons_by_buffer(polygon1, polygon2):
    """Merges 2 polygons by creating a buffer around them so they intersect and appliend a negative buffer after the
    merge"""

    dist = polygon1.distance(polygon2)
    if polygon1.intersects(polygon2):
        geom_union = ops.unary_union([polygon1, polygon2])
        return geom_union

    # Setting the buffer to dist/2 does not guarantee that the 2 polygons will intersect and resulting in
    # a single polygon. Therefore we need the check at the end of the inner loop.
    buffer = dist / 2
    geom1_buf = polygon1.buffer(buffer, resolution=1)
    geom2_buf = polygon2.buffer(buffer, resolution=1)

    if not geom1_buf.intersects(geom2_buf):
        geom_union = geom.MultiPolygon(polygon1, polygon2)
    else:
        geom_union = ops.unary_union([geom1_buf, geom2_buf]).buffer(-buffer, resolution=1)

    return geom_union
