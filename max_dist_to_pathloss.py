import vtovosm as vtv
import numpy as np

# Values from Viriyasitavat et. al.
dist_max_olos_los = 250
dist_max_nlos = 140

ploss = vtv.pathloss.Pathloss()
ploss.disable_shadowfading()

pathloss_max = {}
pathloss_max['olos'] = ploss.pathloss_olos(dist_max_olos_los)
pathloss_max['los'] = ploss.pathloss_los(dist_max_olos_los)
pathloss_max['nlos'] = ploss.pathloss_nlos(dist_max_nlos/2, dist_max_nlos/2)
# pathloss_max['nlos_rx'] = ploss.pathloss_nlos(0.99*dist_max_nlos, 0.01*dist_max_nlos)
# pathloss_max['nlos_tx'] = ploss.pathloss_nlos(0.01*dist_max_nlos, 0.99*dist_max_nlos)

pathloss_max_array = np.array(list(pathloss_max.values()))
pathloss_mean = np.mean(pathloss_max_array)

print(pathloss_max)
print(pathloss_mean)
