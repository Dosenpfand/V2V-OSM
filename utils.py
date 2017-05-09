""" Various uncomplex functionality"""

import sys
import time
import smtplib
from email.mime.text import MIMEText
import socket
import getpass
import gzip
import os
import pickle
import logging
import scipy.special as spc
import scipy.stats as st
import numpy as np


def string_to_filename(string):
    """ Cleans a string up to be used as a filename"""

    keepcharacters = ('_', '-')
    filename = ''.join(c for c in string if c.isalnum()
                       or c in keepcharacters).rstrip()
    filename = filename.lower()
    return filename


def print_nnl(text, file=sys.stdout):
    """Print without adding a new line """

    print(text, file=file, end='', flush=True)


def debug(time_start=None, text=None):
    """ Times execution and outputs log messages"""

    if time_start is None:
        if text is not None:
            logging.info(text)
        time_start = time.process_time()
        return time_start
    else:
        time_diff = time.process_time() - time_start
        logging.debug('Finished in {:.3f} s'.format(time_diff))
        return time_diff


def square2cond(n, i, j):
    """Converts the squareform indices i and j of the condensed vector with size n to the
    condensed index k. See also: scipy.spatial.distance.squareform"""

    k = int(spc.comb(n, 2) - spc.comb(n - i, 2) + (j - i - 1))
    return k


def net_connectivity_stats(net_connectivities, confidence=0.95):
    """Calculates the means and confidence intervals for network connectivity results"""

    means = np.mean(net_connectivities, axis=0)
    conf_intervals = np.zeros([np.size(means), 2])

    for index, mean in enumerate(means):
        conf_intervals[index] = st.t.interval(confidence, len(
            net_connectivities[:, index]) - 1, loc=mean, scale=st.sem(net_connectivities[:, index]))

    return means, conf_intervals


def send_mail_finish(recipient=None, time_start=None):
    """Sends an email to notify someone about the finished simulation"""

    if time_start is None:
        msg = MIMEText('The simulation is finished.')
    else:
        msg = MIMEText('The simulation started at {:.0f} is finished.'.format(
            time_start))

    msg['Subject'] = 'Simulation finished'
    msg['From'] = getpass.getuser() + '@' + socket.getfqdn()
    if recipient is None:
        msg['To'] = msg['From']
    else:
        msg['To'] = recipient

    try:
        smtp = smtplib.SMTP('localhost')
    except ConnectionRefusedError:
        logging.error('Connection to mailserver refused')
    else:
        smtp.send_message(msg)
        smtp.quit()


def save(obj, file_path, protocol=4, compression_level=1):
    """Saves an object using gzip compression"""

    with gzip.open(file_path, 'wb', compresslevel=compression_level) as file:
        pickle.dump(obj, file, protocol=protocol)


def load(file_path):
    """Loads and decompresses a saved object"""

    with gzip.open(file_path, 'rb') as file:
        return pickle.load(file)


def compress_file(file_in_path, protocol=4, compression_level=1, delete_uncompressed=True):
    """Loads an uncompressed file and saves a compressed copy of it"""

    file_out_path = file_in_path + '.gz'
    with open(file_in_path, 'rb') as file_in:
        obj = pickle.load(file_in)
        save(obj, file_out_path, protocol=protocol,
             compression_level=compression_level)

    if delete_uncompressed:
        os.remove(file_in_path)


def fill_config(config):
    """ Set unset SUMO settings to sane defaults"""

    if 'tls_settings' not in config['sumo']:
        config['sumo']['tls_settings'] = None
    if 'fringe_factor' not in config['sumo']:
        config['sumo']['fringe_factor'] = None
    if 'max_speed' not in config['sumo']:
        config['sumo']['max_speed'] = None
    if 'intermediate_points' not in config['sumo']:
        config['sumo']['intermediate_points'] = None
    if 'warmup_duration' not in config['sumo']:
        config['sumo']['warmup_duration'] = None
    if 'abort_after_sumo' not in config['sumo']:
        config['sumo']['abort_after_sumo'] = False
    if 'multiprocess' not in config:
        config['multiprocess'] = False

    return config
