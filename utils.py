""" Various uncomplex functionality"""

import time


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
