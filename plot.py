""" Plot functionality"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import osmnx as ox
import utils

# TODO: define figure and axis for every plot function call?
# TODO: add option to save figure for every function?


def plot_streets_and_buildings(streets, buildings=None, show=True, filename=None, dpi=300):
    """ Plots streets and buildings"""

    # TODO: street width!
    # TODO: bug when plotting buildings, inner area not empty!
    fig, axi = ox.plot_graph(
        streets, show=False, close=False, node_size=0, dpi=dpi, edge_color='#333333')

    if buildings is not None:
        ox.plot_buildings(buildings, fig=fig, ax=axi,
                          show=False, close=False, dpi=dpi, color='#999999')

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    return fig, axi


def plot_prop_cond(streets, buildings, coordinates_vehs, show=True, place=None):
    """ Plots vehicles and their respective propagation condition (LOS/OLOS/NLOS parallel/NLOS
    orthoognal)"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with propagation conditions
    plt.scatter(coordinates_vehs.get('center')[0], coordinates_vehs.get('center')[1], label='Own',
                marker='x', zorder=10, s=2 * plt.rcParams['lines.markersize']**2, c='black')
    plt.scatter(coordinates_vehs.get('los')[:, 0], coordinates_vehs.get('los')[:, 1], label='LOS',
                zorder=9, alpha=0.75)
    plt.scatter(coordinates_vehs.get('olos')[:, 0], coordinates_vehs.get('olos')[:, 1],
                label='OLOS', zorder=8, alpha=0.75)
    plt.scatter(coordinates_vehs.get('orth')[:, 0], coordinates_vehs.get('orth')[:, 1],
                label='NLOS orth', zorder=5, alpha=0.5)
    plt.scatter(coordinates_vehs.get('par')[:, 0], coordinates_vehs.get('par')[:, 1],
                label='NLOS par', zorder=5, alpha=0.5)

    # Add additional information to plot
    plt.legend()
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    title_string = 'Vehicle positions and propagation conditions'
    if place is not None:
        title_string += ' ({})'.format(place)
    plt.title(title_string)

    if show:
        plt.show()

    return fig, axi


def plot_pathloss(streets, buildings, vehicles, show=True, place=None):
    """ Plots vehicles and their respecitive pathloss color coded"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with pathlosses
    pathlosses = vehicles.get_pathlosses('other')
    index_wo_inf = pathlosses != np.Infinity
    index_inf = np.invert(index_wo_inf)
    plt.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Own',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize']**2)
    cax = plt.scatter(vehicles.get('other')[index_wo_inf][:, 0],
                      vehicles.get('other')[index_wo_inf][:, 1], marker='o',
                      c=pathlosses[index_wo_inf], cmap=plt.cm.magma, label='Finite PL')
    plt.scatter(vehicles.get('other')[index_inf][:, 0],
                vehicles.get('other')[index_inf][:, 1], marker='.', c='y',
                label='Infinite PL', alpha=0.5)

    # Add additional information to plot
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()
    title_string = 'Vehicle positions and pathlosses'
    if place is not None:
        title_string += ' ({})'.format(place)
    axi.set_title(title_string)

    # Plot color map
    pl_min = np.min(pathlosses[index_wo_inf])
    pl_max = np.max(pathlosses[index_wo_inf])
    pl_med = np.mean((pl_min, pl_max))
    string_min = '{:.0f}'.format(pl_min)
    string_med = '{:.0f}'.format(pl_med)
    string_max = '{:.0f}'.format(pl_max)
    cbar = fig.colorbar(
        cax, ticks=[pl_min, pl_med, pl_max], orientation='vertical')
    cbar.ax.set_xticklabels([string_min, string_med, string_max])
    cbar.ax.set_xlabel('Pathloss [dB]')

    if show:
        plt.show()

    return fig, axi


def plot_con_status(streets, buildings, coordinates_vehs, show=True, place=None):
    """ Plots the connection status (connected/not conected) in regard to another vehicle"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with connection status
    plt.scatter(coordinates_vehs.get('center')[0], coordinates_vehs.get('center')[1], label='Own',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize']**2, zorder=3)
    plt.scatter(coordinates_vehs.get('in_range')[:, 0], coordinates_vehs.get('in_range')[:, 1],
                label='In range', marker='o', zorder=2)
    plt.scatter(coordinates_vehs.get('out_range')[:, 0], coordinates_vehs.get('out_range')[:, 1],
                label='Out of range', marker='o', alpha=0.75, zorder=1)

    # Add additional information to plot

    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()
    title_string = 'Vehicle positions and connectivity'

    if place is not None:
        title_string += ' ({})'.format(place)
    plt.title(title_string)

    if show:
        plt.show()

    return fig, axi


def plot_cluster_max(streets, buildings, coordinates_vehs, show=True, place=None):
    """ Plots the biggest cluster and the remainding vehicles"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with connection status
    plt.scatter(coordinates_vehs.get('cluster_max')[:, 0],
                coordinates_vehs.get('cluster_max')[:, 1],
                label='Biggest cluster', marker='o', zorder=2)
    plt.scatter(coordinates_vehs.get('not_cluster_max')[:, 0],
                coordinates_vehs.get('not_cluster_max')[:, 1],
                label='Other vehicles', marker='o', alpha=0.75, zorder=1)

    # Add additional information to plot

    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()
    title_string = 'Vehicle positions and biggest cluster'

    if place is not None:
        title_string += ' ({})'.format(place)
    plt.title(title_string)

    if show:
        plt.show()

    return fig, axi


def plot_net_connectivity_comp(filename):
    """ Plots the simulated network connectivity statistics and the ones from the paper"""

    with open(filename, 'rb') as file:
        results = pickle.load(file)
    net_connectivities = results['out']['net_connectivities']

    aver_net_cons, conf_net_cons = utils.net_connectivity_stats(
        net_connectivities)

    aver_net_cons_paper = np.array([12.67, 18.92, 21.33,
                                    34.75, 69.72, 90.05, 97.46, 98.97, 99.84, 100]) / 100
    conf_net_cons_paper = np.array(
        [2.22, 4.51, 2.57, 6.58, 8.02, 3.48, 1.25, 0.61, 0.25, 0]) / 100
    net_densities = np.concatenate([np.arange(10, 90, 10), [120, 160]])

    plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    plt.rc('text', usetex=True)
    plt.errorbar(net_densities, aver_net_cons,
                 np.abs(aver_net_cons - conf_net_cons.T), label='OSM (own method)')
    plt.errorbar(net_densities, aver_net_cons_paper, conf_net_cons_paper,
                 label='Manhattan grid (Viriyasitavat et al.)')

    # Add additional information to plot
    plt.xlabel(r'Network density $[veh/km^2]$')
    plt.ylabel(r'Average network connectivity [\%]')
    plt.legend()
    plt.show()


def plot_veh_traces_animation(traces, streets, buildings=None, show=True, filename=None):
    """Plots an animation of the vehicle traces"""

    # TODO: make whole function prettier

    def update_line(timestep, traces, line):
        """Updates the animation periodically"""
        line.set_data([traces[timestep]['x'], traces[timestep]['y']])
        return line,

    fig, _ = plot_streets_and_buildings(
        streets, buildings=buildings, show=False)
    line, = plt.plot([], [], 'ro')

    line_anim = animation.FuncAnimation(fig, update_line, len(traces), fargs=(traces, line),
                                        interval=25, blit=True)
    if show:
        plt.show()

    if filename is not None:
        writer = animation.writers['ffmpeg']
        writer_inst = writer(fps=15, bitrate=1800)
        line_anim.save(filename, writer=writer_inst)
