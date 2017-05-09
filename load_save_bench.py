import utils
import pickle
import logging
import os

loglevel = logging.getLevelName('DEBUG')
logger = logging.getLogger()
logger.setLevel(loglevel)

data_path = 'results/sumo_upperwestside-newyork-usa.386.pickle'
with open(data_path, 'rb') as file:
    data = pickle.load(file)


levels = range(9)

for level in levels:
    file_path = data_path + str(level) + '.gz'
    time_start = utils.debug(text='Writing and reading level ' + str(level))
    utils.save(data, file_path, compression_level=level)
    utils.debug(time_start, 'Writing')
    time_start = utils.debug()
    data_temp = utils.load(file_path)
    utils.debug(time_start, 'Reading')
    file_size = os.path.getsize(file_path)
    utils.debug(text='Filesize {:.1f} MB'.format(file_size / 1e6))
