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

from g3.molecules import pyg_to_sequence, is_valid_molecule, draw_mol
from g3.utils import pad_2d_tensor

def filter_dataset(dataset, repeats):
    num_samples = 0
    valid_indices = []
    
    for i, graph_data in enumerate(tqdm.tqdm(dataset)):
        seq, adj = pyg_to_sequence(graph_data)
        valid, _, _ = is_valid_molecule(seq, adj, verbose=False)
        
        if valid:
            num_samples += min(repeats, graph_data.num_nodes)
            valid_indices.append(i)
            
    return num_samples, valid_indices
                   
def get_node_token(graph, node_id):
    base = [ (graph.x[node_id][:5]).argmax().item()]
    return base + [0]
   
def tokenize_graph( graph, max_num_nodes, num_tokens ,dtype):
    sequence = np.array(
        [get_node_token(graph, i) for i in range(graph.num_nodes)]
        + [[0] * num_tokens] * (max_num_nodes - graph.num_nodes),
        dtype=dtype
    )
    
    return sequence
   
# starts with a random node and makes preorder dfs
def get_permutation(data, root_node=0, build_alg='dfs'):
    G = to_networkx(data, to_undirected=True)
    
    if build_alg == 'dfs':
        node_order = list(nx.dfs_preorder_nodes(G, source=root_node))
        #logger.info("using dfs")
    elif build_alg == 'bfs':
        #logger.info("using bfs")
        node_order = list(nx.bfs_tree(G, source=root_node).nodes)
    else:
        raise ValueError(f"{build_alg} not supported")
    
    permutation = torch.tensor(node_order).to(torch.int)
    edge_index, edge_attr = subgraph(
        permutation, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True
    )
    permuted_graph = Data(data.x[permutation], edge_index, edge_attr)
    adj_mat = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.long)
    edge_types = torch.argmax(permuted_graph.edge_attr, dim=1) 
    adj_mat[edge_index[0], edge_index[1]] = edge_types + 1
    
    temp = torch.tril(adj_mat + 1, diagonal=-1)
    edge_target = temp[temp != 0] 
    edge_target -= 1
    assert edge_target.shape[0] == data.num_nodes * (data.num_nodes -1) / 2
    
    return permuted_graph, edge_target, adj_mat

def create_dataset(datadir, name, dataset, repeats, batch_size, max_seq_len, num_tokens=2, filter=True, build_alg='dfs'):
    filename = f"{datadir}/{name}.bin"
    logger.info(f"Creating dataset under {filename}")
    
    if filter:
        logger.info(f"Filtering Dataset")
        num_samples, valid_indices = filter_dataset(dataset, repeats)
        logger.info(f"Dataset {name} has {len(dataset) - len(valid_indices)} invalid found")  
        dataset = dataset.index_select(valid_indices)   
        logger.info(f"Dataset {name} has {len(dataset)} data and with repeats {num_samples} samples")    
    else: 
        num_samples = len(dataset)   
         
    dtype = np.uint16  # For small graphs use np.uinit8 to save half memory
    build_shape = (num_samples, max_seq_len, num_tokens)
    logger.info(f"Creating a memmap of size {build_shape} under {filename}")
    build_sequences = np.memmap(filename, dtype=dtype, mode="w+", shape=build_shape)

    batch_idx = 0
    batch = []
    num_nodes_batch_list = []
    special_target_list = []
    edge_type_target_list = []
    attn_bias_list = []
    
    for i, data in enumerate(tqdm.tqdm(dataset)):
        repeat_count = min(repeats, data.num_nodes)
        start_nodes = random.sample(range(0, data.num_nodes), repeat_count) if repeats > 1 else [0]
        
        for root in start_nodes:
            # get permutation with dfs and edge_target(lower tringular adj matrix and 0 for no edge)     
            graph_data, edge_type_target, adj_mat = get_permutation(data, root_node=root, build_alg=build_alg)
            build_sequence = tokenize_graph(graph_data, max_seq_len, num_tokens, dtype)
            assert edge_type_target.max() <= 3
            
            # attention bias
            padded_adj = pad_2d_tensor(adj_mat, max_seq_len - adj_mat.shape[0])
            padded_adj.fill_diagonal_(4)
            
            # degree info
            degree_info = torch.tril(padded_adj, diagonal=-1).sum(-1).numpy()
            build_sequence[:,-1] = degree_info
            if degree_info.max() > 4:
                ag = 10
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
            
            if len(batch) >= batch_size:
                # NOTE: This should ensure that runtime only scales by the time
                # a single permutation takes, not also the number of memory accesses
                build_sequences[batch_idx : batch_idx + batch_size] = np.stack(batch)
                batch_idx += batch_size
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


def prepare_qm9(name, root, vocab, num_train = 100_000, repeats=1, batch_size=1, num_tokens=2, build_alg="dfs", filter=True):
    full_dataset = QM9(root)

    #min_num_nodes = min([data.x.size(0) for data in full_dataset])
    max_num_nodes = 29
    #max_num_edges = max([data.edge_index.size(1) for data in full_dataset])
    max_num_edges = 56
    
    train_dataset = full_dataset[:num_train]
    val_dataset = full_dataset[num_train:]

    if not os.path.exists(datadir := f"{root}/{name}"):
        logger.info(f"Creating new directory for dataset under {datadir}")
        os.makedirs(f"{datadir}")

    
    train_shape = create_dataset(
        datadir = datadir,
        name= "train", 
        dataset= train_dataset, 
        repeats= repeats, 
        batch_size =batch_size, 
        max_seq_len= max_num_nodes, 
        num_tokens=num_tokens,
        filter=filter,
        build_alg = build_alg
    )
    
    val_shape = create_dataset(
        datadir= datadir, 
        name= "val", 
        dataset= val_dataset, 
        repeats= 1, 
        batch_size= batch_size, 
        max_seq_len = max_num_nodes, 
        num_tokens=num_tokens,
        filter=filter,
        build_alg = build_alg
    )

    logger.info("Saving metadata...")
    metadata = dict(
        train_shape=train_shape,
        val_shape=val_shape,
        num_node_type=vocab.node_type,
        num_edge_type=vocab.edge_type,
        num_special_type=vocab.special_type,
        num_degree_type=vocab.degree_type,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        num_targets=full_dataset.num_classes,
    )
    torch.save(metadata, f"{datadir}/metadata.pt")
    logger.info(metadata)