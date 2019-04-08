import pandas as pd
import numpy as np
import networkx as nx
import json


def load_data(edge_path, community_path):
    """
    load graph data, including edges and communities.
    :param edge_path: path of edges (may be a list of paths)
    :param community_path: path of community file (format: {node1:comm_id}, json)
    :return: the graph and community dict
    """
    if isinstance(edge_path, list):
        edge_list = []
        for path in edge_path:
            edge_df_ = pd.read_csv(path, delimiter=" |,")
            edge_list.append(edge_df_.values)
        edge_list = np.vstack(edge_list)
        edge_df = pd.DataFrame(data=edge_list)
    else:
        edge_df = pd.read_csv(edge_path, delimiter=" |,")
    graph = nx.DiGraph()
    graph.add_edges_from(edge_df.iloc[:,:2].values)
    if edge_df.shape[1] == 3 and isinstance(edge_df.iloc[0, 2], float):
        nx.set_edge_attributes(graph,
                               dict(zip(
                                   list(zip(edge_df.iloc[:, 0], edge_df.iloc[:, 1])),
                                   edge_df.iloc[:, 2])),
                               name="weight")
    else:
        nx.set_edge_attributes(graph,
                               dict(zip(
                                   list(zip(edge_df.iloc[:, 0], edge_df.iloc[:, 1])),
                                   np.ones(len(edge_df)))),
                               name="weight")
    if community_path[-4:] == 'json':
        community_dict = json.load(open(community_path, "r"))
    else:
        community_dict = generate_community_dict(graph, communities_path=community_path, save=True)
    nx.set_node_attributes(graph, community_dict, name="community")
    return graph, community_dict


def generate_community_dict(graph, communities_path=None, save=True):
    """
    generate community dict (format: {node1:comm_id}) from a community list, if the path of
    community list is not specified, use greed_modularity to generate.
    :param graph: input graph.
    :param communities_path: path of community list file.
    :param save: whether to save the result (.json).
    :return: generated community dict
    """
    if communities_path == None:
        communities = [list(community) for community in
                       list(nx.algorithms.community.greedy_modularity_communities(graph))]
    else:
        communities = load_communities(communities_path)
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    if save:
        json.dump(community_dict, open("input/community.json", "w"))
    return community_dict


def load_communities(communities_path):
    """
    load community list.
    :param communities_path:
    :return:
    """
    communities = []
    with open(communities_path, "r") as f:
        for line in f.readlines():
            communities.append([int(x) for x in line.split(" ")[:-1]])
    return communities
