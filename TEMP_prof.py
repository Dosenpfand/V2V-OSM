import vtovosm as vtv
import vtovosm.simulations.result_analysis as ra
import time


in_p = 'results/default.200.pickle.xz'
out_p = 'results/TEMP.pickle.xz'


time_s = time.time()
ra.analyze_single(in_p, out_p, ['connection_durations'], multiprocess=True)
time_d = time.time() - time_s
print(time_d)

time_s = time.time()
ra.analyze_single(in_p, out_p, ['connection_durations'], multiprocess=False)
time_d = time.time() - time_s
print(time_d)
