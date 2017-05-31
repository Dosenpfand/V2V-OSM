"""Determine if using a building tolerance makes a difference on the resulting connection matrices"""

import os
import signal

import numpy as np

import main_sim
import network_parser as nw_p
import utils


def analyze_tolerance(conf_path):
    """Analyzes the simulation results by comparing connection matrices from simulations with and without tolerance.
    The connection matrices correspond to propagation condition matrices with True = OLOS/LOS and False = NLOS because
    of the maximum set distances."""

    config = nw_p.params_from_conf(config_file=conf_path)

    if config['density_type'] != 'absolute':
        raise NotImplementedError('Only absolute vehicle counts supported')

    counts_vehs = config['densities_veh']

    scenarios = nw_p.get_scenarios_list(conf_path)
    suffixes = set()
    for scenario in scenarios:
        suffixes.add(scenario[12:])

    for suffix in list(suffixes):

        results = []

        for count_vehs in counts_vehs:
            res_wo = utils.load('results/tolerance_0_{}.{:d}.pickle.gz'.format(suffix, count_vehs))
            res_w = utils.load('results/tolerance_1_{}.{:d}.pickle.gz'.format(suffix, count_vehs))
            run_time_wo = res_wo['info']['time_finish'] - res_wo['info']['time_start']
            run_time_w = res_w['info']['time_finish'] - res_w['info']['time_start']
            matrices_cons_wo = res_wo['results']['matrices_cons']
            matrices_cons_w = res_w['results']['matrices_cons']

            count_diff = 0
            count_tot = 0

            for matrix_cons_wo, matrix_cons_w in zip(matrices_cons_wo, matrices_cons_w):
                count_diff += np.nonzero(matrix_cons_wo != matrix_cons_w)[0].size
                count_tot += matrix_cons_w.size

            ratio_diff = count_diff / count_tot
            result = {'count_vehs': count_vehs,
                      'count_con_tot': count_tot,
                      'count_con_diff': count_diff,
                      'ratio_con_diff': ratio_diff,
                      'run_time_wo': run_time_wo,
                      'run_time_w': run_time_w}
            results.append(result)

        utils.save(results, 'results/tolerance_comparison_{}.pickle.gz'.format(suffix))


if __name__ == '__main__':
    # Set the config to be used
    config_file_path = 'network_config/tolerance_inspection.json'

    # Register signal handler
    signal.signal(signal.SIGTSTP, main_sim.signal_handler)

    # Change to directory of script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main_sim.main_multi_scenario(conf_path=config_file_path)

    results = analyze_tolerance(config_file_path)
    print(results)
