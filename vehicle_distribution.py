""" Distributes vehicles along streets"""

import numpy as np


def choose_random_streets(lengths, count=1):
    """ Chooses random streets with probabilities relative to their length"""
    total_length = sum(lengths)
    probs = lengths / total_length
    count_streets = np.size(lengths)
    indices = np.zeros(count, dtype=int)
    indices = np.random.choice(count_streets, size=count, p=probs)
    return indices


def choose_random_point(street, count=1):
    """Chooses random points along street """
    distances = np.random.random(count)
    points = np.zeros_like(distances, dtype=object)
    for index, dist in np.ndenumerate(distances):
        points[index] = street.interpolate(dist, normalized=True)

    return points
