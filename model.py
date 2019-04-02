import numpy as np
import pandas as pd
import networkx as nx
from functools import reduce, partial
import pathos.multiprocessing as mp
import time
from tqdm import tqdm
from utils import *


class CommunityBasedGreedyAlgorithm:
    def __init__(self, graph, communities, K, average_diffusion_speed):
        """
        Initialize with graph, community_dict and several parameters.
        :param graph: input graph.
        :param community_dict: input community dict.
        :param K: K in "top-K".
        :param average_diffusion_speed: average diffusion speed.
        """
        print("Reading data and performing initialization ...")
        self.K = K
        self.average_diffusion_speed = average_diffusion_speed
        self.graph = graph
        # nx.set_node_attributes(self.graph, community_dict, name='community')
        self.N = len(list(graph.nodes()))
        self.M = len(communities)
        self.communities = communities
        edge_weights = [x[2] for x in graph.edges.data("weight")]
        self.wt_max = np.max(edge_weights)
        self.wt_min = np.min(edge_weights)

        self.I = [[] for i in range(self.M + 1)]
        self.R = np.zeros([self.M + 1, self.K + 1])
        self.s = np.zeros([self.M + 1, self.K + 1])
        self.node_list = []

    def compute_diffusion_speed(self, node1, node2):
        """
        compute the diffusion speed with input node ids.
        :param node1: the first node id.
        :param node2: the second node id.
        :return: diffusion speed from node1 to node2.
        """
        wt = self.graph.edges[node1, node2]['weight']
        diffusion_speed = 2 * self.average_diffusion_speed * wt / (self.wt_max + self.wt_min)
        return diffusion_speed

    def compute_influence_degree(self, node_list, community):
        """
        Compute the influence degree of a list of nodes in a subgraph (identified by community id)
        :param node_list: list of nodes.
        :param community: community id for generating the subgraph (0...M-1).
        :return: influence degree.
        """
        # t0 = time.time()
        node_list = np.asarray(node_list)
        num_nodes = len(node_list)
        if num_nodes==0: return 0
        subgraph = nx.subgraph(self.graph, self.communities[community])
        # print(len(subgraph.nodes))
        signal = True
        mother_node_list = node_list
        # multiprocess_flag = len(subgraph.nodes) > 300
        # if len(subgraph.nodes) > 300:
        #     pool = mp.Pool()

        while signal:
            # if multiprocess_flag:
            #     nodes_to_add = reduce(np.union1d,
            #                           pool.map(lambda x: self.diffuse(subgraph, x), mother_node_list))
            #
            # else:
            nodes_to_add = reduce(np.union1d,
                                  list(map(lambda x: self.diffuse(subgraph, x),
                                           mother_node_list)))
            nodes_to_add = [node for node in nodes_to_add if not node in node_list]
            # exclude nodes already exist in node_list,
            # which could lead to loop(circle) and massive computation.
            if len(nodes_to_add) == 0:
                signal = False
            else:
                node_list = np.union1d(node_list, nodes_to_add)
                mother_node_list = nodes_to_add
        influence_degree = len(np.unique(node_list)) / self.N
        # print(len(subgraph)," graph nodes,",len(node_list),":",time.time() - t0,"s")
        return influence_degree

    def diffuse(self, graph, source_node):
        """
        One step of generating new nodes from a source node.
        :param graph: subgraph.
        :param source_node: one single node.
        :return:
        """
        if not source_node in graph.nodes: return np.array([])
        nbrs = np.array(list(graph.successors(source_node)))
        if len(nbrs) == 0: return []
        random_nums = np.random.random([len(nbrs)])
        diffusion_speeds = np.array(list(map(
            lambda x: self.compute_diffusion_speed(source_node, x),
            nbrs)))
        flags = random_nums <= diffusion_speeds
        nodes_to_add_ = nbrs[flags]
        return nodes_to_add_

    def optimize(self):
        """
        perform dynamic programming algorithm.
        :return: None.
        """
        print("Optimizing ...")
        t0 = time.time()
        # pool = mp.Pool()
        for k in tqdm(range(1,self.K+1)):
            for m in tqdm(range(1,self.M+1)):
                nodes = self.communities[m - 1]
                # print(nodes)
                R = self.compute_influence_degree(self.I[0], m-1)

                delta_R_list = list(map(
                    lambda x: self.compute_influence_degree(self.I[0]+[x], m-1), nodes))

                # delta_R_list = pool.map(partial(compute_influence_degree, community=m-1, model=self),
                #                         [self.I[0] + [x] for x in nodes])
                # delta_R_list = [a.get() for a in delta_R_list]
                delta_R = np.max(delta_R_list) - R
                self.R[m, k] = max(self.R[m-1,k], self.R[self.M,k-1] + delta_R)
                if self.R[m-1, k] >= self.R[self.M, k-1] + delta_R:
                    self.s[m, k] = self.s[m-1, k]
                else:
                    self.s[m, k] = m

            j = int(self.s[self.M, k])
            R1 = self.compute_influence_degree(self.I[j], j-1)
            delta_R_list1 = list(map(lambda x: self.compute_influence_degree(self.I[j] + [x], j-1),
                             self.communities[j-1]))
            # delta_R_list1 = [a.get() for a in delta_R_list1]
            v_max = self.communities[j-1][np.argmax(delta_R_list1)]
            self.I[j].append(v_max)
            self.I[0].append(v_max)
            # Unsure whether the v_max could have appeared in self.I[j] or self.I[0]
        # pool.close()
        # pool.join()
        print("Done, cost", time.time() - t0, "s.")


# A test of model
if __name__ == "__main__":

    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    graph = nx.algorithms.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5,
                                                        min_community=20, seed=10)
    graph = nx.DiGraph(graph)
    community_dict = generate_community_dict(graph, 20)
    weight_dict = dict(zip(graph.edges, np.random.random(graph.number_of_edges())))
    nx.set_edge_attributes(graph, weight_dict, name = "weight")
    model = CommunityBasedGreedyAlgorithm(graph, community_dict, 10, 0.2)
    model.optimize()
    print(model.I[0])