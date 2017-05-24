"""Determine if using a building tolerance makes a difference on the resulting connection matrices"""

import os
import signal

import numpy as np

import main_sim
import utils


def analyze_tolerance():
    """Analyzes the simulation results by comparing connection matrices from simulations with and without tolerance"""

    counts_vehs = [10, 100, 1000]

    results = []

    for count_vehs in counts_vehs:
        res_wo = utils.load('results/tolerance_test_without.{:d}.pickle.gz'.format(count_vehs))
        res_w = utils.load('results/tolerance_test_with.{:d}.pickle.gz'.format(count_vehs))
        matrices_cons_wo = res_wo['results']['matrices_cons']
        matrices_cons_w = res_w['results']['matrices_cons']

        count_diff = 0
        count_tot = 0

        for matrix_cons_wo, matrix_cons_w in zip(matrices_cons_wo, matrices_cons_w):
            count_diff += np.nonzero(matrix_cons_wo != matrix_cons_w)[0].size
            count_tot += matrix_cons_w.size

        ratio_diff = count_diff / count_tot
        result = {'count_vehs': count_vehs, 'count_tot': count_tot, 'count_diff': count_diff, 'ratio_diff': ratio_diff}
        results.append(result)

    utils.save(results, 'results/tolerance_test_comparison')
    return results


if __name__ == '__main__':

    # Register signal handler
    signal.signal(signal.SIGTSTP, main_sim.signal_handler)

    # Change to directory of script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    scenarios = ['tolerance_test_with', 'tolerance_test_without']

    for scenario in scenarios:
        # Run main function
        main_sim.main(scenario=scenario)

    results = analyze_tolerance()
    print(results)
