# cga
Python implementation of Community-based Graph Algorithm for finding top-K nodes with most influences in a graph

Python = 3.5 with numpy, pandas, networkx, pathos and tqdm.

Input data:

1. one whole graph or multiple subgraphs(based on community) in format of:
    an edge list with source, target, weight (separated by space)
2. community list in format of:
    a text file where each line is a list of nodes for a community.

Output:

1. a txt file with first line of top K nodes and second line of influence degree.

