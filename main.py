import numpy as np
import pandas as pd
import argparse
from model import CommunityBasedGreedyAlgorithm
from utils import *


def arg_parser():
    """
    Generate an arguments parser.
    :return:
    """
    parser = argparse.ArgumentParser(description="All arguments have their default values.")
    parser.add_argument("--mode", type=int, default=0,
                        help="Switch between single whole graph and multiple subgraphs.")
    parser.add_argument("--input_path", type=str, default="input/facebook.txt",
                        help="The path of input graph.")
    parser.add_argument("--input_prefix", type=str, default="input/level",
                        help="The prefix used to generate edge file list.")
    parser.add_argument("--communities_path", type=str, default="input/levels",
                        help="The path of community file, only used if the list of communities exists.")
    parser.add_argument("--head_tail", type=list, default=[1,9],
                        help="The head and tail of names in edge file list.")
    parser.add_argument("--output_path", type=str, default="output/topK.txt",
                        help="The path of result.")
    parser.add_argument("--K", type=int, default=10,
                        help="The K in topK.")
    parser.add_argument("--average_diffusion_speed", type=float, default=0.01,
                        help="The average diffusion speed, " +
                             "which should be lower than 1/(average degree of whole graph).")
    return parser


def main(args):
    if args.mode == 0:
        graph, community_dict = load_data(args.input_path, args.communities_path)
    else:
        edge_file_list = [args.input_prefix + str(i) + ".txt" \
                          for i in range(args.head_tail[0], args.head_tail[1] + 1)]
        graph, community_dict = load_data(edge_file_list, args.communities_path)
    communities = load_communities(args.communities_path)
    # overlapping exists, communities generated from community_dict is wrong, use the original communities file.
    model = CommunityBasedGreedyAlgorithm(graph, communities, args.K, args.average_diffusion_speed)
    model.optimize()
    topK_nodes = model.I[0]
    print(topK_nodes)
    with open(args.output_path, "w") as f:
        for node in topK_nodes:
            f.write(str(node) + " ")
        f.write("\n")
        f.write(str(model.R[-1,-1]))


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args)