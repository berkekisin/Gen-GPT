import os
import random
import networkx as nx
import torch
import tqdm
from loguru import logger
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx, subgraph, dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
from networkx import grid_2d_graph

from g3.utils import pad_2d_tensor
   
# starts with a random node and makes preorder dfs
def get_permutation(data, root_node=1, build_alg='dfs'):
   
    node_mapping = {node: i for i, node in enumerate(sorted(data.nodes()))}
    data = nx.relabel_nodes(data, node_mapping)
    
    if build_alg == 'dfs':
        node_order = list(nx.dfs_preorder_nodes(data, source=0))
    elif build_alg == 'bfs':
        node_order = list(nx.bfs_tree(data, source=root_node).nodes)
    else:
        raise ValueError(f"{build_alg} not supported")
    
    edge_list = list(data.edges())
    old_edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)
    edge_attr = torch.zeros(old_edge_index.shape[1]).to(torch.long)
    node_count = len(node_order)
    node_x = torch.zeros(node_count)
    
    permutation = torch.tensor(node_order).to(torch.int)
    edge_index, edge_attr = subgraph(
        permutation, old_edge_index, edge_attr=edge_attr, relabel_nodes=True
    )
    permuted_data = Data(node_x, edge_index, edge_attr)
    adj_mat = torch.zeros((node_count, node_count), dtype=torch.long)
    adj_mat[edge_index[0], edge_index[1]] = edge_attr + 1
    adj_mat_s = torch.max(adj_mat, adj_mat.T)

    
    temp = torch.tril(adj_mat_s + 1, diagonal=-1)
    edge_target = temp[temp != 0] 
    edge_target -= 1
    assert edge_target.shape[0] == node_count * (node_count -1) / 2
    
    return permuted_data, edge_target, adj_mat_s

def create_dataset(dataset, cfg, name='train', datadir =None):
    filename = f"{datadir}/{name}.bin"
    logger.info(f"Creating dataset under {filename}")
    max_seq_len = cfg.dataset.range.max ** 2
    
    repeats = cfg.dataset.repeats
    num_samples = len(dataset) * repeats
    build_shape = (num_samples, max_seq_len, cfg.dataset.num_tokens)
    logger.info(f"Creating a memmap of size {build_shape} under {filename}")
    
    with open(filename, "wb") as f:
        f.seek(np.prod(build_shape) * np.dtype(np.uint16).itemsize - 1)  # Allocate space
        f.write(b'\x00')  # Write a single byte to set the file size

    build_sequences = np.memmap(filename, dtype=np.uint16 , mode="w+", shape=build_shape)

    batch_idx = 0
    batch = []
    num_nodes_batch_list = []
    special_target_list = []
    edge_type_target_list = []
    attn_bias_list = []
    
    for i, data in enumerate(tqdm.tqdm(dataset)):
        repeat_count = min(repeats, data.number_of_nodes())
        start_nodes = random.sample(range(0, data.number_of_nodes()), repeat_count) if repeats > 1 else [0]
        
        for root in start_nodes:
            # get permutation with dfs and edge_target(lower tringular adj matrix and 0 for no edge)     
            graph_data, edge_type_target, adj_mat = get_permutation(
                data, 
                root_node=root, 
                build_alg=cfg.dataset.build_algorithm
            )
            build_sequence = np.zeros(shape=(max_seq_len,2),dtype=np.uint16)
            
            # attention bias
            padded_adj = pad_2d_tensor(adj_mat, max_seq_len - adj_mat.shape[0])
            padded_adj.fill_diagonal_(cfg.dataset.vocab.edge_type-1)
            
            # degree info
            degree_info = torch.tril(padded_adj, diagonal=-1).sum(-1).numpy()
            build_sequence[:,-1] = degree_info
            assert degree_info.max() <= 4
     
            #special target
            special_target = torch.ones(size=(graph_data.num_nodes,)).to(torch.int64)
            special_target[-1] = 0
            
            # append to list
            batch.append(build_sequence)
            num_nodes_batch_list.append(graph_data.num_nodes)
            attn_bias_list.append(padded_adj)
            special_target_list.append(special_target)
            edge_type_target_list.append(edge_type_target)
            
            if len(batch) >= cfg.dataset.batch_size:
                # NOTE: This should ensure that runtime only scales by the time
                # a single permutation takes, not also the number of memory accesses
                build_sequences[batch_idx : batch_idx + cfg.dataset.batch_size] = np.stack(batch)
                batch_idx += cfg.dataset.batch_size
                batch = []

    if len(batch) > 0:
        logger.info(f"Last batch had size {len(batch)}")
        build_sequences[batch_idx:] = np.stack(batch)

    logger.info(f"Done generating {num_samples} build sequences")

    build_sequences.flush()
    torch.save(
        dict(
            num_nodes_batch = num_nodes_batch_list,
            attn_bias = attn_bias_list,
            special_target = special_target_list,
            edge_type_target = edge_type_target_list
        ),
        f"{datadir}/{name}.pkl",
    )

    return build_shape

def get_dataset(low, high):
    dataset = []
    
    for i in range(low, high):
        for j in range(low, high):
            dataset.append(grid_2d_graph(i,j).to_undirected())
    
    return dataset
    

def prepare_grids(cfg):
    dataset_name = f'{cfg.dataset.name}-r{cfg.dataset.repeats}-{cfg.dataset.build_algorithm}'
    datadir = f"{cfg.root}/{dataset_name}"
    
    full_dataset = get_dataset(low=cfg.dataset.range.min, high=cfg.dataset.range.max)
    
    if not os.path.exists(datadir):
        logger.info(f"Creating new directory for dataset under {datadir}")
        os.makedirs(f"{datadir}")
        
    num_train = len(full_dataset) - cfg.dataset.num_val

    train_dataset = full_dataset[:num_train]
    val_dataset = full_dataset[ num_train:]
    
    train_shape = create_dataset(train_dataset, cfg, name='train', datadir=datadir)
    val_shape = create_dataset(val_dataset, cfg, name='val', datadir=datadir)
    
    max_num_nodes = cfg.dataset.range.max ** 2
    logger.info("Saving metadata...")
    metadata = dict(
        train_shape=train_shape,
        val_shape=val_shape,
        num_node_type=cfg.dataset.vocab.node_type,
        num_edge_type=cfg.dataset.vocab.edge_type,
        num_special_type=cfg.dataset.vocab.special_type,
        num_degree_type=cfg.dataset.vocab.degree_type,
        max_num_nodes=max_num_nodes
    )
    torch.save(metadata, f"{datadir}/metadata.pt")
    logger.info(metadata)
    
    
    
if __name__ == "__main__":
    get_permutation('aa')