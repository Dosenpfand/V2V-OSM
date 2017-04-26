import time
import requests

import osmnx

# TODO: clean up whole file!

def osm_net_download(polygon, network_type='all_private', timeout=180, memory=None, max_query_area_size=50*1000*50*1000):

    # create a filter to exclude certain kinds of ways based on the requested network_type
    osm_filter = osmnx.get_osm_filter(network_type)
    response_jsons = []

    # pass server memory allocation in bytes for the query to the API
    # if None, pass nothing so the server will use its default allocation size
    # otherwise, define the query's maxsize parameter value as whatever the caller passed in
    if memory is None:
        maxsize = ''
    else:
        maxsize = '[maxsize:{}]'.format(memory)


    # project to utm, divide polygon up into sub-polygons if area exceeds a max size (in meters), project back to lat-long, then get a list of polygon(s) exterior coordinates
    geometry_proj, crs_proj = osmnx.project_geometry(polygon)
    geometry_proj_consolidated_subdivided = osmnx.consolidate_subdivide_geometry(geometry_proj, max_query_area_size=max_query_area_size)
    geometry, crs = osmnx.project_geometry(geometry_proj_consolidated_subdivided, crs=crs_proj, to_latlong=True)
    polygon_coord_strs = osmnx.get_polygons_coordinates(geometry)
    # TODO: !
    # log('Requesting network data within polygon from API in {:,} request(s)'.format(len(polygon_coord_strs)))
    # start_time = time.time()

    # pass each polygon exterior coordinates in the list to the API, one at a time
    for polygon_coord_str in polygon_coord_strs:
        query_template = '[out:xml][timeout:{timeout}]{maxsize};(way["highway"]{filters}(poly:"{polygon}");>;);out;'
        query_str = query_template.format(polygon=polygon_coord_str, filters=osm_filter, timeout=timeout, maxsize=maxsize)
        response_json = overpass_request(data={'data':query_str}, timeout=timeout)
        response_jsons.append(response_json)
    # log('Got all network data within polygon from API in {:,} request(s) and {:,.2f} seconds'.format(len(polygon_coord_strs), time.time()-start_time))

    return response_jsons

def overpass_request(data, pause_duration=None, timeout=180, error_pause_duration=None):

    # define the Overpass API URL, then construct a GET-style URL as a string to hash to look up/save to cache
    url = 'http://www.overpass-api.de/api/interpreter'
    prepared_url = requests.Request('GET', url, params=data).prepare().url
    # TODO: !
    # cached_response_json = get_from_cache(prepared_url)
    cached_response_json = None

    if not cached_response_json is None:
        # found this request in the cache, just return it instead of making a new HTTP call
        return cached_response_json

    else:
        # if this URL is not already in the cache, pause, then request it
        if pause_duration is None:
            this_pause_duration = osmnx.get_pause_duration()
        # TODO: !
        # log('Pausing {:,.2f} seconds before making API POST request'.format(this_pause_duration))
        time.sleep(this_pause_duration)
        start_time = time.time()
        # TODO: !
        # log('Posting to {} with timeout={}, "{}"'.format(url, timeout, data))
        response = requests.post(url, data=data, timeout=timeout)

        # get the response size and the domain, log result
        # TODO: !
        # size_kb = len(response.content) / 1000.
        # domain = re.findall(r'//(?s)(.*?)/', url)[0]
        # log('Downloaded {:,.1f}KB from {} in {:,.2f} seconds'.format(size_kb, domain, time.time()-start_time))

        if response.status_code in [429, 504]:
            # pause for error_pause_duration seconds before re-trying request
            if error_pause_duration is None:
                error_pause_duration = osmnx.get_pause_duration()
            # TODO: !
            # log('Server at {} returned status code {} and no JSON data. Re-trying request in {:.2f} seconds.'.format(domain, response.status_code, error_pause_duration), level=lg.WARNING)
            time.sleep(error_pause_duration)
            response_json = overpass_request(data=data, pause_duration=pause_duration, timeout=timeout)

        return response.content
