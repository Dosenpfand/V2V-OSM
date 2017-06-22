import logging
import multiprocessing as mp
import networkx as nx

from vtovosm import connection_analysis as con_ana
from vtovosm import utils

loglevel = logging.getLevelName('INFO')
logger = logging.getLogger()
logger.setLevel(loglevel)

filepath_res = 'results/viriyasitavat_comparison/result.771.pickle.xz'
filepath_ana = 'results/viriyasitavat_comparison/TEMP_ANA.pickle.xz'
config_analysis = ['net_connectivities']

results_loaded = utils.load(filepath_res)

matrices_cons = results_loaded['results']['matrices_cons']
processes = 16

graphs_cons = []
for matrix_cons in matrices_cons:
    graphs_cons.append(nx.from_numpy_matrix(matrix_cons))
    print('generated graph')

with mp.Pool(processes=processes) as pool:
    net_connectivities = pool.map(con_ana.calc_net_connectivity, graphs_cons)

utils.save(net_connectivities, filepath_ana)
