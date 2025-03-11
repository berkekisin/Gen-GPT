import networkx as nx
import torch

from torch_geometric.utils import to_networkx

def compute_shortest_path(data, max_seq_len):
    graph_networkx = to_networkx(data, ["x"], ["edge_attr"], to_undirected=True)
    dist_dict = dict(nx.all_pairs_shortest_path_length(graph_networkx))
    n = len(dist_dict)
    dist_tensor = torch.zeros((max_seq_len, max_seq_len))

    for i in range(n):
        for j in range(n):
            dist_tensor[i, j] = dist_dict[i][j]
            
    return dist_tensor


def compute_rw_pe(adj, walk_length):
    curr = adj
    pe_list = [curr]
    for _ in range(walk_length - 1):
        curr = curr @ adj
        pe_list.append(curr)
        
    return pe_list