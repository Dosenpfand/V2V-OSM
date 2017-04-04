""" Plot functionality"""

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

# TODO: define figure and axis for every plot function call?
# TODO: add option to save figure for every function?
# TODO: change indices? 0 for vehicle, 1 for x/y


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
    fig, axi = plot_streets_and_buildings(streets, buildings, show=False, dpi=300)

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
    fig, axi = plot_streets_and_buildings(streets, buildings, show=False, dpi=300)

    # Plot vehicles with pathlosses
    pathlosses = vehicles.get_pathlosses('other')
    index_wo_inf = pathlosses != np.Infinity
    index_inf = np.invert(index_wo_inf)
    plt.scatter(vehicles.get('center')[0], vehicles.get('center')[1], label='Own',
                c='black', marker='x', s=2 * plt.rcParams['lines.markersize']**2)
    cax = plt.scatter(vehicles.get('other')[index_wo_inf][:, 0],
                      vehicles.get('other')[index_wo_inf][:, 1], marker='o', \
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
    cbar = fig.colorbar(cax, ticks=[pl_min, pl_med, pl_max], orientation='vertical')
    cbar.ax.set_xticklabels([string_min, string_med, string_max])
    cbar.ax.set_xlabel('Pathloss [dB]')

    if show:
        plt.show()

    return fig, axi


def plot_con_status(streets, buildings, coordinates_vehs, show=True, place=None):
    """ Plots the connection status (connected/not conected) in regard to another vehicle"""

    # Plot streets and buildings
    fig, axi = plot_streets_and_buildings(streets, buildings, show=False, dpi=300)

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
