import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import vtovosm as vtv

file_format = 'pdf'
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams['text.usetex'] = True
# plt.rcParams['lines.linewidth'] = 3
plt.rcParams['pgf.rcfonts'] = False
plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.style.use('ggplot')

# Street network an graph generation
place = 'Salmannsdorf - Vienna - Austria'
G = ox.graph_from_place(place, network_type='drive')
boundary = ox.gdf_from_place(place)
polygon = boundary['geometry'].iloc[0]
buildings = ox.create_buildings_gdf(polygon)

G_ren = nx.convert_node_labels_to_integers(G)
G_pro = ox.project_graph(G_ren)

fig, axi = ox.plot_graph(G_pro, node_color='#66ccff', edge_color='#999999', node_size=120, annotate=True, show=False)
fig.tight_layout(pad=0)
fig.savefig('images/framework/street_network_0.{}'.format(file_format))

fig, axi = plt.subplots()
nx.draw_networkx(G_pro, ax=axi, pos=nx.spring_layout(G_pro), node_color='#66ccff', edge_color='#999999', node_size=200)
axi.set_axis_off()
fig.tight_layout(pad=0)
fig.savefig('images/framework/street_network_1.{}'.format(file_format))

# Building simplification
places = ['Salmannsdorf - Vienna - Austria', 'Neubau - Vienna - Austria', 'Upper West Side - New York - USA']
for place in places:
    place_filename = vtv.utils.string_to_filename(place)

    boundary = ox.gdf_from_place(place)
    polygon = boundary['geometry'].iloc[0]
    build = ox.create_buildings_gdf(polygon)
    build_proj = ox.project_gdf(build)
    build_simp = vtv.osmnx_addons.simplify_buildings(build_proj)

    fig, axi = ox.plot_buildings(build_proj, show=False)
    fig.tight_layout(pad=0)
    fig.savefig('images/framework/simplify_buildings_0_' + place_filename + '.{}'.format(file_format))
    fig, axi = ox.plot_buildings(build_simp, show=False)
    fig.tight_layout(pad=0)
    fig.savefig('images/framework/simplify_buildings_1_' + place_filename + '.{}'.format(file_format))

    edge_count_simp = 0
    for geom in build_simp.geometry:
        edge_count_simp += len(geom.exterior.coords)

    edge_count = 0
    for geom in build_proj.geometry:
        edge_count += len(geom.exterior.coords)

    print(place)
    print('Simplified:')
    print('Number of polygons:', len(build_simp))
    print('Total number of edges:', edge_count_simp)
    print('Original:')
    print('Number of polygons:', len(build_proj))
    print('Total number of edges:', edge_count)
    print()
