try:
    import matplotlib
    import matplotlib.pyplot as plt
    import networkx as nx

    matplotlib.use('qt5agg')
except ImportError:
    matplotlib = None
    plt = None
    nx = None


def vis_map_graph(mat, nodes, edges, graph_name):
    fig = plt.figure(figsize=(21, 15))
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edges)
    for n, p in nodes.items():
        G.nodes[n]['pos'] = p

    plt.imshow(mat, cmap='Pastel1')

    plt.tight_layout()

    # to see labels, set with_labels=True
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=True, font_size=10, node_size=200,
            node_shape='s', alpha=0.8)

    figManager = plt.get_current_fig_manager()
    figManager.window.move(-3840, 0)
    figManager.window.showMaximized()
    fig.canvas.start_event_loop(0.001)

    plt.savefig('output/%s/%s_string.png' % (graph_name, graph_name))
    plt.show()

    fig = plt.figure(figsize=(21, 15))
    figManager = plt.get_current_fig_manager()
    figManager.window.move(-3840, 0)
    figManager.window.showMaximized()
    fig.canvas.start_event_loop(0.001)
    G = nx.convert_node_labels_to_integers(G)
    plt.cla()
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=True, font_size=10, node_size=200,
            node_shape='s', alpha=0.8)
    plt.imshow(mat, cmap='Pastel1')
    plt.savefig('output/%s/%s.png' % (graph_name, graph_name))
    plt.show()

    return G
