"""Geometrical functionality"""

import shapely.ops as ops
import numpy as np


def line_intersects_buildings(line, buildings):
    """Returns `True` if `line` intersects with any of the `buildings`.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Geometrical line
    buildings : geopandas.GeoDataFrame
        Buildings inside a geodata frame

    Returns
    -------
    intersects : bool
        True if `line` intersects buildings, otherwise false
    """
    """ """

    intersects = False
    for geometry in buildings.geometry:
        if line.intersects(geometry):
            intersects = True
            break

    return intersects


def line_intersects_points(line, points, margin=1):
    """Returns `True` if `line` intersects with any of the `points` within a `margin`.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Geometrical line
    points : list of shapely.geometry.Point
        A list of geometrical points
    margin : float
        The maximum margin between `line` and `points`

    Returns
    -------
    intersects : bool
        True if `line` intersects `points`, otherwise false

    """

    intersects = False

    for point in points:
        circle = point.buffer(margin)
        intersects = circle.intersects(line)
        if intersects:
            break

    return intersects


def get_street_lengths(graph_streets):
    """Returns the lengths of the streets in `graph_streets`.

    Parameters
    ----------
    graph_streets : networkx.MultiDiGraph
        Street network graph generated by osmnx

    Returns
    -------
    lengths : float
        sum of length of all streets

    Notes
    -----
    There are small differences in the values of data['geometry'].length
    and data['length']
    """

    lengths = np.zeros(graph_streets.number_of_edges())
    for index, street in enumerate(graph_streets.edges_iter(data=True)):
        lengths[index] = street[2]['length']
    return lengths


def extract_point_array(points):
    """Extracts coordinates form a point array

    Parameters
    ----------
    points : list of shapely.geometry.Point

    Returns
    -------
    coords : numpy.ndarray
        Coordinates
    """

    coords = np.zeros([np.size(points), 2], dtype=float)

    for index, point in np.ndenumerate(points):
        coords[index, :] = np.transpose(point.xy)

    return coords


def find_center_veh(coords):
    """Finds the index of the centermost coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates

    Returns
    -------
    index_center_veh : int
        Index of the centermost vehicle

    """

    min_x = np.amin(coords[:, 0])
    max_x = np.amax(coords[:, 0])
    min_y = np.amin(coords[:, 1])
    max_y = np.amax(coords[:, 1])
    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2
    coords_center = np.array((mean_x, mean_y))
    distances_center = np.linalg.norm(
        coords_center - coords, ord=2, axis=1)
    index_center_veh = np.argmin(distances_center)
    return index_center_veh


def split_line_at_point(line, point):
    """Splits the `line` at the `point` on the line and returns it's two parts.

    Parameters
    ----------
    line : shapely.geometry.LineString
    point : shapely.geometry.Point

    Returns
    -------
    line_before: shapely.geometry.LineString
        Part of `line` before `point`
    line_after: shapely.geometry.LineString
        Part of `line` after `point`

    """

    if line.distance(point) > 1e-8:
        raise ValueError('Point not on line')

    # Use small buffer polygon instead of point to deal with floating point precision
    circle = point.buffer(1e-8)
    line_split = ops.split(line, circle)
    line_before = line_split[0]
    line_after = line_split[-1]

    return line_before, line_after


def angles_along_line(line):
    """Determines the the `angles` along the `line` string.
    For a line consisting of n segments the function returns n-1 angles where the i-th element is the angle between
    segment i and i+1.

    Parameters
    ----------
    line : shapely.geometry.LineString

    Returns
    -------
    angles : numpy.ndarray
    """

    coords = line.coords
    angles = np.zeros(len(coords) - 2)
    angle_temp_prev = 0

    for index, coord in enumerate(coords[1:]):
        coord_prev = coords[index]
        angle_temp = np.arctan2(
            coord[0] - coord_prev[0], coord[1] - coord_prev[1])
        if index != 0:
            if angle_temp - angle_temp_prev < np.pi:
                angles[index - 1] = angle_temp - angle_temp_prev + np.pi
            else:
                angles[index - 1] = angle_temp - angle_temp_prev - np.pi
        angle_temp_prev = angle_temp

    return angles


def wrap_to_pi(angle):
    """Limits angle from -pi to +pi.

    Parameters
    ----------
    angle : float

    Returns
    -------
    wrapped_angle : float
    """

    return (angle + np.pi) % (2 * np.pi) - np.pi
