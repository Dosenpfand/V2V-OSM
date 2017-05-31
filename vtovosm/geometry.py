""" Geometrical functionality"""

import shapely.ops as ops
import numpy as np


def line_intersects_buildings(line, buildings):
    """ Checks if a line intersects with any of the buildings"""

    intersects = False
    for geometry in buildings.geometry:
        if line.intersects(geometry):
            intersects = True
            break

    return intersects


def line_intersects_points(line, points, margin=1):
    """Checks if a line intersects with any of the points within a margin"""

    intersects = False

    for point in points:
        circle = point.buffer(margin)
        intersects = circle.intersects(line)
        if intersects:
            break

    return intersects


def get_street_lengths(streets):
    """Returns the lengths of the streets in a graph"""

    # NOTE: The are small differences in the values of data['geometry'].length
    # and data['length']
    lengths = np.zeros(streets.number_of_edges())
    for index, street in enumerate(streets.edges_iter(data=True)):
        lengths[index] = street[2]['length']
    return lengths


def extract_point_array(points):
    """Extracts coordinates form a point array """

    coords = np.zeros([np.size(points), 2], dtype=float)

    for index, point in np.ndenumerate(points):
        coords[index, :] = np.transpose(point.xy)

    return coords


def find_center_veh(coords):
    """Finds the index of the vehicle at the center of the map """

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
    """Splits a line at the point on the line """

    if line.distance(point) > 1e-8:
        raise ValueError('Point not on line')

    # NOTE: Use small circle instead of point to get around floating point
    # precision
    circle = point.buffer(1e-8)
    line_split = ops.split(line, circle)
    line_before = line_split[0]
    line_after = line_split[-1]

    return line_before, line_after


def angles_along_line(line):
    """Determines the the angles along a line"""

    coord_prev = []
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
    """ Limits angle from -pi to +pi"""

    return (angle + np.pi) % (2 * np.pi) - np.pi
