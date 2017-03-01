
import numpy as np
from matplotlib import pyplot as plt
import time
np.random.seed(int(time.time()))
lam_r = 1e-3
lam_v = 1e-2
road_len = 5000
crange = 1000
reps = 1000
distri = np.zeros((crange, reps))
for rep in range(reps):
    print(rep)
    if lam_r * road_len < 1:
        n_x = 1
    else:
        n_x = 1 + np.random.poisson(lam_r * road_len - 1, 1)
    n_y = np.max((0, np.random.poisson(lam_r * road_len, 1)))

    xcoords = np.random.uniform(0, road_len, n_x)
    ycoords = np.random.uniform(0, road_len, n_y)

    veh_nr_x = np.random.poisson(lam_v * road_len, n_x)

    veh_nr_y = np.random.poisson(lam_v * road_len, n_y)
    veh_coords = np.zeros((np.sum(veh_nr_y) + np.sum(veh_nr_x), 2))
    idx = 0
    for i, n in enumerate(veh_nr_x):
        veh_coords[idx:idx + veh_nr_x[i],
                   1] = np.random.uniform(0, road_len, veh_nr_x[i])
        veh_coords[idx:idx + veh_nr_x[i], 0] = xcoords[i]
        idx += veh_nr_x[i]
    for i, n in enumerate(veh_nr_y):
        veh_coords[idx:idx + veh_nr_y[i],
                   0] = np.random.uniform(0, road_len, veh_nr_y[i])
        veh_coords[idx:idx + veh_nr_y[i], 1] = ycoords[i]
        idx += veh_nr_y[i]
    c_coords = road_len / 2 * np.ones_like(veh_coords)
    d = np.sum((veh_coords - c_coords)**2, 1)

    centroid = np.array([veh_coords[np.argmin(d), :]])
    c_col = np.ones((len(veh_coords), 1)) @ centroid
    # c_col = c_coords
    dn = np.sort(np.sum((veh_coords - c_col)**2, 1))

    d_x = np.arange(crange)
    for i in d_x:
        distri[i, rep] = np.count_nonzero(dn <= i**2) - 1
plt.figure()
plt.scatter(veh_coords[:, 0], veh_coords[:, 1])
plt.figure()
plt.plot(d_x, np.mean(distri, 1))
r_cor = 2 * lam_r * d_x / \
    (1 - (1 * np.exp(- 2 * lam_r * d_x))) - 1
r_cor[np.where(r_cor < 0)] = 0
# corrected[np.where(corrected < 0)] = 0
v_cor = lam_v * 2 * d_x / (1 - (1 * np.exp(- 2 * lam_v * d_x))) - 1
#
N_l = 2 * d_x * lam_v / (1 - np.exp(-2 * d_x * lam_v)) - 1
N_p = 2 *  (( d_x * lam_r) / (1 - np.exp(- d_x * lam_r)) - 1) * \
    np.pi / 2 * lam_v * d_x
N_c = lam_r * 2 * d_x * np.pi / 2 * d_x * lam_v

f = N_l + N_p + N_c
# f2 = v_cor + r_cor * np.pi / 2 * lam_v * d_x + lam_v * lam_r * np.pi * d_x**2
plt.plot(d_x, f)
# plt.plot(d_x, f2)
plt.figure()
plt.plot(d_x, (np.mean(distri, 1) - f) / 1)
plt.grid(True)
plt.figure()
plt.plot(d_x, (np.mean(distri, 1) / f))
plt.grid(True)
plt.show()