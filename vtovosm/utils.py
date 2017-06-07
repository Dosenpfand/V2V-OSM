"""Various functionality that does not fit in any other module."""

import datetime
import getpass
import gzip
import logging
import os
import pickle
import smtplib
import socket
import sys
import time
from email.mime.text import MIMEText

import numpy as np
import scipy.stats as st


def string_to_filename(string):
    """Returns a cleaned up string that can be used as a filename.

    Parameters
    ----------
    string : str
        String that will be cleaned up.

    Returns
    -------
    filename: str
        Cleaned up string
    """

    keepcharacters = ('_', '-')
    filename = ''.join(c for c in string if c.isalnum()
                       or c in keepcharacters).rstrip()
    filename = filename.lower()
    return filename


def seconds_to_string(seconds):
    """Converts an amount of seconds to a a string with format "dd:hh:mm:ss".

    Parameters
    ----------
    seconds : int or float
        Number of seconds that will be converted

    Returns
    -------
    string : str
        Formatted string
    """

    dtime = datetime.datetime(
        1, 1, 1) + datetime.timedelta(seconds=int(seconds))

    string = '{:02d}:{:02d}:{:02d}:{:02d}'.format(
        dtime.day - 1,
        dtime.hour,
        dtime.minute,
        dtime.second)

    return string


def print_nnl(text, file=sys.stdout):
    """Print without adding a new line.

    Parameters
    ----------
    text : str
        Text to be printed
    file: optional
        file object to which will be printed
    """

    print(text, file=file, end='', flush=True)


def debug(time_start=None, text=None):
    """Times execution and outputs log messages.
    If `time_start` is None, it will be interpreted as a start message and `text` will be logged.
    If `time_start` is None, it will be interpreted as start time and `text` will be logged together with the time
    difference between now and `time_start`.

    Parameters
    ----------
    time_start : float, optional
        Start time of the corresponding action.
    text : str, optional
        Text that will be logged
    """

    if time_start is None:
        if text is not None:
            logging.info(text)
        time_start = time.process_time()
        return time_start
    else:
        time_diff = time.process_time() - time_start
        logging.debug('Finished in {:.3f} s'.format(time_diff))
        return time_diff


def square_to_condensed(idx_i, idx_j, size_n):
    """Converts the squareform indices i and j of the square matrix with with size `size_n` x `size_n` to the
    condensed index k.

    Parameters
    ----------
    idx_i : int
        Row index of the square matrix
    idx_j : int
        Column index of the square matrix
    size_n :
        Size of the square matrix

    Returns
    -------
    k : int
        Index of the condensed vector

    See Also
    --------
    scipy.spatial.distance.squareform
    """

    if idx_i == idx_j:
        raise ValueError('Diagonal entries are not defined')
    if idx_i < idx_j:
        idx_i, idx_j = idx_j, idx_i
    k = size_n * idx_j - idx_j * (idx_j + 1) / 2 + idx_i - 1 - idx_j
    return int(k)


def condensed_to_square(index_k, size_n):
    """Converts the condensed index k of the condensed vector to the indicies i and j of the square matrix with
    size `size_n` x `size_n`.

    Parameters
    ----------
    index_k : int
        Index of the condensed vector
    size_n : int
        Size of the square matrix

    Returns
    -------
    i : int
        Row index of the square matrix
    j : int
        Column index of the square matrix

    See Also
    --------
    scipy.spatial.distance.squareform
    """

    def calc_row_idx(index_k, size_n):
        """Determines the row index"""
        return int(
            np.ceil((1 / 2.) *
                    (- (-8 * index_k + 4 * size_n ** 2 - 4 * size_n - 7) ** 0.5
                     + 2 * size_n - 1) - 1))

    def elem_in_i_rows(index_i, size_n):
        """Determines the number of elements in the i-th row"""
        return index_i * (size_n - 1 - index_i) + (index_i * (index_i + 1)) / 2

    def calc_col_idx(index_k, index_i, size_n):
        """Determines the column index"""
        return int(size_n - elem_in_i_rows(index_i + 1, size_n) + index_k)

    i = calc_row_idx(index_k, size_n)
    j = calc_col_idx(index_k, i, size_n)

    return i, j


def net_connectivity_stats(net_connectivities, confidence=0.95):
    """Calculates the means and confidence intervals for network connectivity results.

    Parameters
    ----------
    net_connectivities : list of float
        Network connectivities
    confidence : float, optional
        Confidence interval

    Returns
    -------
    means : array of float
        Means
    conf_intervals : array of float
        Confidence intervals

    """

    means = np.mean(net_connectivities, axis=0)
    conf_intervals = np.zeros([np.size(means), 2])

    for index, mean in enumerate(means):
        conf_intervals[index] = st.t.interval(confidence, len(
            net_connectivities[:, index]) - 1, loc=mean, scale=st.sem(net_connectivities[:, index]))

    return means, conf_intervals


def send_mail_finish(recipient=None, time_start=None):
    """Sends an email to notify someone about the finished simulation using a local mail server.

    Parameters
    ----------
    recipient : str, optional
        recipient of the email. If it is `None` than it will be sent to the user executing python.
    time_start: float, optional
        Time at which the simulation has been started.
    """

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


def save(obj, file_path, protocol=4, compression_level=1, overwrite=True, create_dir=True):
    """Saves an object using gzip compression.

    Parameters
    ----------
    obj :
        Object that will be saved
    file_path : str
        Path at which `obj` will be saved
    protocol : int, optinal
        Gzip protocol
    compression_level : int, optional
        Gzip compression level
    overwrite : bool, optional
        When true an already existing file will be overwritten.
    create_dir : bool, optional
        When true any non existing intermediary directories in `file_path` will be created.
    """

    # Return if file already exists
    if not overwrite and os.path.isfile(file_path):
        return

    # Create the output directory if it does not exist
    if create_dir:
        directory = os.path.dirname(file_path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

    with gzip.open(file_path, 'wb', compresslevel=compression_level) as file:
        pickle.dump(obj, file, protocol=protocol)


def load(file_path):
    """Loads and decompresses a saved object.

    Parameters
    ----------
    file_path : str
        Path of the compressed file

    Returns
    -------
    object
        Object that was saved in the file
    """

    with gzip.open(file_path, 'rb') as file:
        return pickle.load(file)


def compress_file(file_in_path, protocol=4, compression_level=1, delete_uncompressed=True):
    """Loads an uncompressed file and saves a compressed copy of it

    Parameters
    ----------
    file_in_path : str
        Path of the uncompressed file
    protocol : int, optinal
        Gzip protocol version
    compression_level : int, optional
        Gzip compression level
    delete_uncompressed : bool, optional
        When True, the uncompressed file will be deleted after compression
    """

    file_out_path = file_in_path + '.gz'
    with open(file_in_path, 'rb') as file_in:
        obj = pickle.load(file_in)
        save(obj, file_out_path, protocol=protocol,
             compression_level=compression_level)

    if delete_uncompressed:
        os.remove(file_in_path)
