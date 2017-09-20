""" Plot functionality"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox


def setup(figsize=(8, 5)):
    """Sets up plotting"""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["savefig.bbox"] = 'tight'

    plt.rcParams['text.usetex'] = True
    plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})


def plot_streets_and_buildings(streets, buildings=None, show=True, dpi=300, path=None, overwrite=False, ruler=True,
                               axes=False):
    """ Plots streets and buildings"""

    fig, axi = ox.plot_graph(
        streets, show=False, close=False, node_size=0, dpi=dpi, edge_color='#333333', fig_height=6)

    # TODO: bug when plotting buildings, inner area not empty! (e.g. Stiftskaserne Wien Neubau)
    if buildings is not None:
        ox.plot_buildings(buildings, fig=fig, ax=axi, set_bounds=False, show=False, close=False, dpi=dpi,
                          color='#999999')

    # Reset axes parameters to default
    if axes:
        axes_color = '#999999'
        axi.axis('on')
        axi.margins(0.05)
        axi.tick_params(which='both', direction='out', colors=axes_color)
        axi.set_xlabel('X coordinate [m]', color=axes_color)
        axi.set_ylabel('Y coordinate [m]', color=axes_color)
        axi.spines['right'].set_color('none')
        axi.spines['top'].set_color('none')
        axi.spines['left'].set_color(axes_color)
        axi.spines['bottom'].set_color(axes_color)
        fig.canvas.draw()

    if ruler:
        plot_ruler(axi)

    if path is not None:
        if overwrite or not os.path.isfile(path):
            fig.savefig(path)

    if show:
        fig.show()

    return fig, axi


def plot_vehs(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """Plots vehicles"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with propagation conditions
    axi.scatter(vehicles.get()[:, 0], vehicles.get()[:, 1])

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
    axi.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Center',
                marker='x', zorder=10, s=2 * plt.rcParams['lines.markersize'] ** 2, c='black')
    axi.scatter(vehicles.get('los')[:, 0], vehicles.get('los')[:, 1], label='LOS',
                zorder=9, alpha=0.75)
    axi.scatter(vehicles.get('olos')[:, 0], vehicles.get('olos')[:, 1],
                label='OLOS', zorder=8, alpha=0.75)
    axi.scatter(vehicles.get('ort')[:, 0], vehicles.get('ort')[:, 1],
                label='NLOS orth', zorder=5, alpha=0.5)
    axi.scatter(vehicles.get('par')[:, 0], vehicles.get('par')[:, 1],
                label='NLOS par', zorder=5, alpha=0.5)

    # Add additional information to plot
    axi.legend().set_visible(True)

    if path is not None:
        if overwrite or not os.path.isfile(path):
            fig.savefig(path)

    if show:
        fig.show()

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
    axi.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Center',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize'] ** 2, zorder=3)
    cax = plt.scatter(vehicles.get('other')[index_wo_inf][:, 0],
                      vehicles.get('other')[index_wo_inf][:, 1], marker='o',
                      c=pathlosses[index_wo_inf], cmap=plt.cm.magma, label='Finite PL', zorder=2)
    axi.scatter(vehicles.get('other')[index_inf][:, 0],
                vehicles.get('other')[index_inf][:, 1], marker='.', c='y',
                label='Infinite PL', alpha=0.5, zorder=1)

    # Add additional information to plot
    axi.legend().set_visible(True)

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

    return fig, axi, cax


def plot_con_status(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots the connection status (connected/not conected) in regard to another vehicle"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with connection status
    axi.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Center',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize'] ** 2, zorder=3)
    axi.scatter(vehicles.get('in_range')[:, 0], vehicles.get('in_range')[:, 1],
                label='In range', marker='o', zorder=2)
    axi.scatter(vehicles.get('out_range')[:, 0], vehicles.get('out_range')[:, 1],
                label='Out of range', marker='o', alpha=0.75, zorder=1)

    # Add additional information to plot
    axi.legend().set_visible(True)

    if path is not None:
        if overwrite or not os.path.isfile(path):
            fig.savefig(path)

    if show:
        fig.show()

    return fig, axi


def plot_cluster_max(streets, buildings, vehicles, show=True, path=None, overwrite=False):
    """ Plots the biggest cluster and the remaining vehicles"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(
        streets, buildings, show=False, dpi=300)

    # Plot vehicles with connection status
    axi.scatter(vehicles.get('cluster_max')[:, 0],
                vehicles.get('cluster_max')[:, 1],
                label='Biggest cluster', marker='o', zorder=2)
    axi.scatter(vehicles.get('not_cluster_max')[:, 0],
                vehicles.get('not_cluster_max')[:, 1],
                label='Other vehicles', marker='o', alpha=0.75, zorder=1)

    # Add additional information to plot
    axi.legend().set_visible(True)

    if path is not None:
        if overwrite or not os.path.isfile(path):
            fig.savefig(path)

    if show:
        fig.show()

    return fig, axi


def plot_veh_traces_animation(traces, streets, buildings=None, show=True, path=None, overwrite=False):
    """Plots an animation of the vehicle traces"""

    def update_line(timestep, traces, line):
        """Updates the animation periodically"""

        line.set_data([traces[timestep]['x'], traces[timestep]['y']])
        return line,

    fig, axi = plot_streets_and_buildings(
        streets, buildings=buildings, show=False)
    line, = axi.plot([], [], linewidth=0, marker='o')

    line_anim = animation.FuncAnimation(fig, update_line, len(traces), fargs=(traces, line),
                                        interval=25, blit=True)
    if show:
        fig.show()

    if path is not None:
        if overwrite or not os.path.isfile(path):
            if os.path.splitext(path)[1] == '.mp4':
                writer = animation.writers['ffmpeg']
                writer_inst = writer(fps=25, bitrate=1800)
            elif os.path.splitext(path)[1] == '.gif':
                writer = animation.writers['imagemagick']
                writer_inst = writer(fps=12, bitrate=-1)
            else:
                raise RuntimeError('File extension not supported')

            line_anim.save(path, writer=writer_inst)


def plot_ruler(axi, length=1000, coord=None, linewidth=3, color='#999999'):
    """Plots a ruler"""

    if coord is None:
        xlim = axi.get_xlim()
        ylim = axi.get_ylim()
        coord = (xlim[0] + 10, ylim[0] - 50)

    axi.plot([coord[0], coord[0] + length], [coord[1], coord[1]], color=color, linewidth=linewidth)

    axi.text(coord[0] + length / 2, coord[1] + 1, '{:d} m'.format(length), horizontalalignment='center',
             verticalalignment='bottom', color=color)

    axi.autoscale()
