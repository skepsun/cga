import pandas as pd
import numpy as np
import networkx as nx
import json

def load_data(edge_path, community_path):
    edge_df = pd.read_csv(edge_path)
    graph = nx.from_edgelist(edge_df.iloc[:, :2].values)
    nx.set_edge_attributes(graph,
                           dict(zip(
                               list(zip(edge_df.iloc[:, 0], edge_df.iloc[:, 1])),
                               edge_df.iloc[:, 2])),
                           name="weight")
    community_dict = json.load(open(community_path, "r"))
    nx.set_node_attributes(graph, community_dict, name="community")
    return graph, community_dict

def generate_community_file(graph, M, save=True):
    communities = [list(community) for community in list(nx.algorithms.community.greedy_modularity_communities(graph))]
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    if save:
        json.dump(community_dict, open("community.json", "w"))
    return community_dict


class CommunityBasedGreedyAlgorithm:
    def __init__(self, graph, community_dict, K, average_diffusion_speed):
        print("Reading data and performing initialization ...")


        self.K = K
        self.average_diffusion_speed = average_diffusion_speed
        self.graph = graph
        nx.set_node_attributes(self.graph, community_dict, name='community')
        self.N = len(list(graph.nodes()))
        self.M = len(np.unique(list(community_dict.values())))
        self.communities = [[node for node in self.graph.nodes \
                             if self.graph.nodes[node]['community'] == i] \
                            for i in range(self.M)]
        self.wt_max = np.max(list(graph.edges.data('weight')))
        self.wt_min = np.min(list(graph.edges.data('weight')))

        self.I = [[] for i in range(self.M + 1)]
        self.R = np.zeros([self.M + 1, self.K + 1])
        self.s = np.zeros([self.M + 1, self.K + 1])
        self.node_list = []

    def compute_diffusion_speed(self, node1, node2):
        wt = self.graph.edges[node1, node2]['weight']
        diffusion_speed = 2 * self.average_diffusion_speed * wt / (self.wt_max + self.wt_min)
        return diffusion_speed

    def compute_influence_degree(self, node_list, community):
        num_nodes = len(node_list)
        if num_nodes==0: return 0
        subgraph = nx.subgraph(self.graph, [node for node in self.graph.nodes if self.graph.nodes[node]['community'] == community])
        signal = True
        while signal:
            nodes_to_add = []
            for node in node_list:
                if not node in subgraph.nodes: continue
                nbrs = list(subgraph.successors(node))
                if len(nbrs) == 0: continue
                random_nums = np.random.random([len(nbrs)])
                diffusion_speeds = np.array([self.compute_diffusion_speed(node, nbr) for nbr in nbrs])
                flags = random_nums <= diffusion_speeds
                nodes_to_add_ = np.array(nbrs)[flags]
                if len(nodes_to_add) == 0: continue
                else:
                    nodes_to_add.append(nodes_to_add_)
            if len(nodes_to_add) == 0:
                signal = False
            else:
                node_list = node_list + nodes_to_add
        influence_degree = len(np.unique(node_list)) / self.N
        return influence_degree

    def optimize(self):
        print("Optimizing ...")
        for k in range(1,self.K+1):
            for m in range(1,self.M+1):
                nodes = self.communities[m - 1]
                print(nodes)
                delta_R = np.max(
                    [self.compute_influence_degree(self.I[0]+[node], m - 1) \
                     - self.compute_influence_degree(self.I[0], m - 1)
                     for node in nodes])
                self.R[m, k] = max(self.R[m-1,k], self.R[self.M,k-1] + delta_R)
                if self.R[m-1, k] >= self.R[self.M, k-1] + delta_R:
                    self.s[m, k] = self.s[m-1, k]
                else:
                    self.s[m, k] = m

            j = int(self.s[self.M, k])
            v_max = self.communities[j-1][np.argmax([self.compute_influence_degree(self.I[j] + [node], j-1) \
                               - self.compute_influence_degree(self.I[j], j-1)
                               for node in self.communities[j-1]])]
            self.I[j].append(v_max)
            self.I[0].append(v_max)
            # Unsure whether the v_max could have appeared in self.I[j] or self.I[0]


if __name__ == "__main__":

    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    graph = nx.algorithms.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5,
                                                        min_community=20, seed=10)
    graph = nx.DiGraph(graph)
    community_dict = generate_community_file(graph, 20)
    weight_dict = dict(zip(graph.edges, np.random.random(graph.number_of_edges())))
    nx.set_edge_attributes(graph, weight_dict, name = "weight")
    model = CommunityBasedGreedyAlgorithm(graph, community_dict, 10, 0.8)
    model.optimize()