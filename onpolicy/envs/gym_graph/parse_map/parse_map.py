import argparse
import json
import os
from typing import Dict, List, Set

import networkx as nx
import numpy as np

import vis

PLOT_COLOR_VAL = {'RED': 97, 'GREEN': 56, 'BLUE': 17, 'default': 0, 'door': 74, 'air': 129,
                  'other': 0, 'victim': 0}

MAP_XMIN = -2225
MAP_YMIN = 60
MAP_ZMIN = -11
MAP_XMAX = -2086
MAP_YMAX = 62
MAP_ZMAX = 129


def modify_grid(mat, bounds, type):
    for id in bounds:
        min_coor, max_coor = bounds[id]['coordinates']
        for z in range(min_coor['z'], max_coor['z']):
            for x in range(min_coor['x'], max_coor['x']):
                mat[z - MAP_ZMIN, x - MAP_XMIN] = PLOT_COLOR_VAL[type]
    return mat


def read_map(map_params):
    with open('map_data.txt') as f:
        for d in f:
            data = json.loads(d)
        locations = data['locations']
        connections = data['connections']
        loc_bounds = data['loc_bounds']
        con_bounds = data['con_bounds']
        child_locations = data['child_locations']

    mat = np.ones((MAP_ZMAX - MAP_ZMIN, MAP_XMAX - MAP_XMIN)) * PLOT_COLOR_VAL['other']
    mat = modify_grid(mat, loc_bounds, 'air')
    mat = modify_grid(mat, con_bounds, 'air')
    mat[0, 0] = 0
    mat = mat[map_params['bbox']['Z'][0]:
              map_params['bbox']['Z'][1], map_params['bbox']['X'][0]:map_params['bbox']['X'][1]]

    return mat, locations, connections, loc_bounds, con_bounds, child_locations


def make_loc_graph(connections, loc_bounds, map_params):
    nodes = {}
    edges = []
    neighbors = {}

    min_x, max_x = map_params['bbox']['X']
    min_z, max_z = map_params['bbox']['Z']

    for node in loc_bounds:  # both hallway/room parts and doors are nodes
        min_coordinates, max_coordinates = loc_bounds[node]['coordinates']
        z = (min_coordinates['z'] + max_coordinates['z']) / 2. - MAP_ZMIN
        x = (min_coordinates['x'] + max_coordinates['x']) / 2. - MAP_XMIN
        nodes[node] = [x - min_x, z - min_z]  # to correspond to grid representation
        neighbors[node] = set()

    for node in set(nodes.keys()):
        if (nodes[node][1] < 0 or (max_z is not None and nodes[node][1] >= (max_z - min_z))
                or nodes[node][0] < 0 or (max_x is not None and nodes[node][0] >= (max_x - min_x))):
            del nodes[node]
            del neighbors[node]

    for edge in connections:
        # if connections[edge]['type'] == 'extension':
        #     continue
        for i in connections[edge]['connected_locations']:
            for j in connections[edge]['connected_locations']:
                if i != j and i in nodes and j in nodes:
                    edges.append([i, j])
                    neighbors[i].add(j)

    return union(nodes, edges, neighbors, map_params['union'])


def group(nodes, edges, neighbors, root: str, children: Set[str]):
    if root in children:
        children.remove(root)

    for c in children:
        del nodes[c]
        del neighbors[c]
        new_edges = []
        for edge in edges:
            if c in edge:
                if root not in edge:
                    edge[edge.index(c)] = root
                    new_edges.append(edge)
            else:
                new_edges.append(edge)
        edges = new_edges

        for node in neighbors:
            if c in neighbors[node]:
                neighbors[node].remove(c)
                if root != node:
                    neighbors[node].add(root)

    return nodes, edges, neighbors


def union(nodes: Dict, edges: List, neighbors: Dict, union_dict: Dict):
    for k, v in union_dict.items():
        nodes, edges, neighbors = group(nodes, edges, neighbors, k, v)
    return nodes, edges, neighbors


def save_graph(G, mat, graph_name):
    np.save('output/%s/%s.npy' % (graph_name, graph_name), mat)
    nx.write_adjlist(G, 'output/%s/%s.adjlist' % (graph_name, graph_name))
    with open('output/%s/%s_Pos.json' % (graph_name, graph_name), 'w') as f:
        json.dump(nx.get_node_attributes(G, 'pos'), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', default='SaturnRight')
    args = parser.parse_args()

    with open('map_config/' + args.map_name + '_Config.json') as f:
        map_params = json.load(f)

    if not os.path.exists(os.path.join('output', args.map_name)):
        os.makedirs(os.path.join('output', args.map_name))

    mat, locations, connections, loc_bounds, con_bounds, child_locations = read_map(map_params)
    nodes, edges, neighbors = make_loc_graph(connections, loc_bounds, map_params)
    G = vis.vis_map_graph(mat, nodes, edges, args.map_name)
    save_graph(G, mat, args.map_name)
