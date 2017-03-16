import osmnx as ox
import pickle

# Part of Manhattan
g_manh_part = ox.graph_from_bbox(40.77, 40.74, -73.97, -74.00, network_type='drive')
pickle.dump(g_manh_part, open('graphs/manhattan_part.pickle', 'wb'))
g_manh_part_proj = ox.project_graph(g_manh_part)
ox.plot_graph(g_manh_part_proj)

# Vienna
g_vienna = ox.graph_from_place('Vienna, Austria', which_result=2)
pickle.dump(g_vienna, open('graphs/vienna.pickle', 'wb'))
g_vienna_bike = ox.graph_from_place('Vienna', network_type='bike', which_result=2)
pickle.dump(g_vienna_bike, open('graphs/vienna_bike.pickle', 'wb'))
g_vienna_drive = ox.graph_from_place('Vienna', network_type='drive', which_result=2)
pickle.dump(g_vienna_drive, open('graphs/vienna_drive.pickle', 'wb'))
g_vienna_part = ox.graph_from_place('Ottakring, Vienna, Austria', which_result=2)
pickle.dump(g_vienna_part, open('graphs/vienna_part.pickle', 'wb'))

# Bozen
g_bozen = ox.graph_from_place('Bozen, Italy')
pickle.dump(g_bozen, open('graphs/bozen.pickle', 'wb'))
