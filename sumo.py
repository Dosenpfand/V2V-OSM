"""Interface to SUMO – Simulation of Urban MObility, sumo.dlr.de"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_veh_traces(filename):
    """Parses a SUMO traces XML file and returns a numpy array"""
    tree = ET.parse(filename)
    root = tree.getroot()

    traces = np.zeros(len(root), dtype=object)
    for idx_timestep, timestep in enumerate(root):
        trace_timestep = np.zeros(
            len(timestep),
            dtype=[('time', 'float'),
                   ('id', 'uint'),
                   ('x', 'float'),
                   ('y', 'float')])
        for idx_veh_node, veh_node in enumerate(timestep):
            veh = veh_node.attrib
            veh_id = int(veh['id'][3:])
            trace_timestep[idx_veh_node]['time'] = timestep.attrib['time']
            trace_timestep[idx_veh_node]['id'] = veh_id
            trace_timestep[idx_veh_node]['x'] = float(veh['x'])
            trace_timestep[idx_veh_node]['y'] = float(veh['y'])

        traces[idx_timestep] = trace_timestep
    return traces


def min_max_coords(traces):
    """Determines the min and max x and y coordinates of all vehicle traces"""

    # TODO: better performance? set to 0?
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    for trace in traces:
        if np.size(trace) == 0:
            continue
        x_min_iter = trace['x'].min()
        x_max_iter = trace['x'].max()
        y_min_iter = trace['y'].min()
        y_max_iter = trace['y'].max()
        if x_min_iter < x_min:
            x_min = x_min_iter
        if x_max_iter > x_max:
            x_max = x_max_iter
        if y_min_iter < y_min:
            y_min = y_min_iter
        if y_max_iter > y_max:
            y_max = y_max_iter

    return x_min, x_max, y_min, y_max


def plot_veh_traces(traces):
    """Plots an animation of the vehicle traces"""

    # TODO: make whole function prettier
    x_min, x_max, y_min, y_max = min_max_coords(traces)

    def update_line(timestep, traces, line):
        line.set_data([traces[timestep]['x'], traces[timestep]['y']])
        print(timestep, traces[timestep]['x'][0], traces[timestep]['y'][0])
        return line,

    fig1 = plt.figure()
    line, = plt.plot([], [], 'ro')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    line_ani = animation.FuncAnimation(fig1, update_line, len(traces), fargs=(traces, line),
                                       interval=25, blit=True)
    plt.show()


if __name__ == '__main__':
    veh_traces = parse_veh_traces('sumo_traces/neubau_vienna_austria_many.xml')
    plot_veh_traces(veh_traces)
