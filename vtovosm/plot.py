""" Plot functionality"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox


# TODO: define figure and axis?

def show():
    """Shows the figure(s)"""

    plt.show()


def setup(figsize=(8, 5)):
    """Sets up plotting"""

    plt.rcParams["figure.figsize"] = figsize


def plot_streets_and_buildings(streets, buildings=None, show=True, dpi=300, path=None, overwrite=False):
    """ Plots streets and buildings"""

    # TODO: street width!
    # TODO: bug when plotting buildings, inner area not empty!
    fig, axi = ox.plot_graph(
        streets, show=False, close=False, node_size=0, dpi=dpi, edge_color='#333333')

    if buildings is not None:
        ox.plot_buildings(buildings, fig=fig, ax=axi,
                          show=False, close=False, dpi=dpi, color='#999999')

    if path is not None:
        if overwrite or not os.path.isfile(path):
            plt.savefig(path)

    if show:
        plt.show()

    return fig, axi

def plot_vehs(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots vehicles"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with propagation conditions
    plt.scatter(vehicles.get()[:, 0], vehicles.get()[:, 1])

    # Add additional information to plot
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')

    if path is not None:
        if overwrite or not os.path.isfile(path):
            plt.savefig(path)

    if show:
        plt.show()

    return fig, axi

def plot_prop_cond(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots vehicles and their respective propagation condition (LOS/OLOS/NLOS parallel/NLOS
    orthogonal)"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with propagation conditions
    plt.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Own',
                marker='x', zorder=10, s=2 * plt.rcParams['lines.markersize'] ** 2, c='black')
    plt.scatter(vehicles.get('los')[:, 0], vehicles.get('los')[:, 1], label='LOS',
                zorder=9, alpha=0.75)
    plt.scatter(vehicles.get('olos')[:, 0], vehicles.get('olos')[:, 1],
                label='OLOS', zorder=8, alpha=0.75)
    plt.scatter(vehicles.get('ort')[:, 0], vehicles.get('ort')[:, 1],
                label='NLOS orth', zorder=5, alpha=0.5)
    plt.scatter(vehicles.get('par')[:, 0], vehicles.get('par')[:, 1],
                label='NLOS par', zorder=5, alpha=0.5)

    # Add additional information to plot
    plt.legend()
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')

    if path is not None:
        if overwrite or not os.path.isfile(path):
            plt.savefig(path)

    if show:
        plt.show()

    return fig, axi


def plot_pathloss(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots vehicles and their respecitive pathloss color coded"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with pathlosses
    pathlosses = vehicles.get_pathlosses('other')
    index_wo_inf = pathlosses != np.Infinity
    index_inf = np.invert(index_wo_inf)
    plt.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Own',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize'] ** 2)
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

    if path is not None:
        if overwrite or not os.path.isfile(path):
            plt.savefig(path)

    if show:
        plt.show()

    return fig, axi


def plot_con_status(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots the connection status (connected/not conected) in regard to another vehicle"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with connection status
    plt.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Own',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize'] ** 2, zorder=3)
    plt.scatter(vehicles.get('in_range')[:, 0], vehicles.get('in_range')[:, 1],
                label='In range', marker='o', zorder=2)
    plt.scatter(vehicles.get('out_range')[:, 0], vehicles.get('out_range')[:, 1],
                label='Out of range', marker='o', alpha=0.75, zorder=1)

    # Add additional information to plot

    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()

    if path is not None:
        if overwrite or not os.path.isfile(path):
            plt.savefig(path)

    if show:
        plt.show()

    return fig, axi


def plot_cluster_max(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots the biggest cluster and the remaining vehicles"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with connection status
    plt.scatter(vehicles.get('cluster_max')[:, 0],
                vehicles.get('cluster_max')[:, 1],
                label='Biggest cluster', marker='o', zorder=2)
    plt.scatter(vehicles.get('not_cluster_max')[:, 0],
                vehicles.get('not_cluster_max')[:, 1],
                label='Other vehicles', marker='o', alpha=0.75, zorder=1)

    # Add additional information to plot

    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend()

    if path is not None:
        if overwrite or not os.path.isfile(path):
            plt.savefig(path)

    if show:
        plt.show()

    return fig, axi


def plot_veh_traces_animation(traces, streets, buildings=None, show=True, path=None, overwrite=False):
    """Plots an animation of the vehicle traces"""

    def update_line(timestep, traces, line):
        """Updates the animation periodically"""

        line.set_data([traces[timestep]['x'], traces[timestep]['y']])
        return line,

    fig, _ = plot_streets_and_buildings(
        streets, buildings=buildings, show=False)
    line, = plt.plot([], [], linewidth=0, marker='o')

    line_anim = animation.FuncAnimation(fig, update_line, len(traces), fargs=(traces, line),
                                        interval=25, blit=True)
    if show:
        plt.show()

    if path is not None:
        if overwrite or not os.path.isfile(path):
            writer = animation.writers['ffmpeg']
            writer_inst = writer(fps=25, bitrate=1800)
            line_anim.save(path, writer=writer_inst)
