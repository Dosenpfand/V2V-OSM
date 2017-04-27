"""Get street networks from OpenStreetMap in XML format"""

import time
import requests
import osmnx


def osm_net_download(polygon,
                     network_type='all_private',
                     timeout=180,
                     memory=None,
                     max_query_area_size=50 * 1000 * 50 * 1000):
    """Download OSM ways and nodes within a polygon from the Overpass API"""

    osm_filter = osmnx.get_osm_filter(network_type)
    response_xmls = []

    if memory is None:
        maxsize = ''
    else:
        maxsize = '[maxsize:{}]'.format(memory)

    geometry_proj, crs_proj = osmnx.project_geometry(polygon)
    geometry_proj_cons_subdiv = osmnx.consolidate_subdivide_geometry(
        geometry_proj, max_query_area_size=max_query_area_size)
    geometry, _ = osmnx.project_geometry(
        geometry_proj_cons_subdiv, crs=crs_proj, to_latlong=True)
    polygon_coord_strs = osmnx.get_polygons_coordinates(geometry)

    for polygon_coord_str in polygon_coord_strs:
        query_template = \
            '[out:xml][timeout:{timeout}]{maxsize};' + \
            '(way["highway"]{filters}(poly:"{polygon}");>;);out;'
        query_str = query_template.format(
            polygon=polygon_coord_str, filters=osm_filter, timeout=timeout, maxsize=maxsize)
        response_xml = overpass_request(
            data={'data': query_str}, timeout=timeout)
        response_xmls.append(response_xml)
    return response_xmls


def overpass_request(data, pause_duration=None, timeout=180, error_pause_duration=None):
    """Send a request to the Overpass API via HTTP POST and return the XML response"""

    url = 'http://www.overpass-api.de/api/interpreter'
    if pause_duration is None:
        this_pause_duration = osmnx.get_pause_duration()
    time.sleep(this_pause_duration)
    response = requests.post(url, data=data, timeout=timeout)

    if response.status_code in [429, 504]:
        if error_pause_duration is None:
            error_pause_duration = osmnx.get_pause_duration()
        time.sleep(error_pause_duration)
        response = overpass_request(
            data=data, pause_duration=pause_duration, timeout=timeout)

    return response.content
