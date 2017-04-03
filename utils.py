""" Various uncomplex functionality"""

import time
import argparse


def string_to_filename(string):
    """ Cleans a string up to be used as a filename"""
    keepcharacters = ('_', '-')
    filename = ''.join(c for c in string if c.isalnum()
                       or c in keepcharacters).rstrip()
    filename = filename.lower()
    return filename


def print_nnl(text):
    """Print without adding a new line """
    print(text, end='', flush=True)

def debug(is_debug_mode=False, time_start=None, text=None):
    """ Times execution and outputs log messages"""
    if not is_debug_mode:
        return

    if time_start is None:
        if text is not None:
            print_nnl(text)
        time_start = time.process_time()
        return time_start
    else:
        time_diff = time.process_time() - time_start
        print_nnl(': {:.3f} seconds\n'.format(time_diff))
        return time_diff

def parse_arguments():
    """Parses the command line arguments and returns them """
    parser = argparse.ArgumentParser(description='Simulate vehicle connections on map')
    parser.add_argument('-p', type=str, default='Neubau - Vienna - Austria', help='place')
    parser.add_argument('-w', type=int, default=1, help='which result')
    parser.add_argument('-d', type=float, default=1000, help='vehicle density')
    parser.add_argument('-s', type=int, default=1,
                        help='use pathloss for connections (not distance)')
    parser.add_argument('-l', type=float, default=150, help='pathloss threshold [dB]')
    parser.add_argument('-t', type=str, default='absolute',
                        help='density type (absolute, length, area)')
    arguments = parser.parse_args()
    return arguments
