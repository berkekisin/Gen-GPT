import random
import hydra
import torch
import numpy as np
from loguru import logger
import os

from g3.io import load_inference_model, load_metadata
from g3.device import device_setup, DeviceInfo
from g3.molecules import is_valid_molecule, draw_mol, save_mol_pickle

import networkx as nx
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = False


@hydra.main(version_base=None, config_path="./config/", config_name="generate")
def main(cfg):
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    logger.info(f"Initial seed: {torch.initial_seed()}")
    device_info = device_setup(cfg.root, d=cfg.device)
    
    logger.info("Loading inference Model")
    model = load_inference_model(
        cfg.root, cfg.checkpoint, cfg.compile, device_info.device, edge_attn=cfg.edge_attn
    )
    
    logger.info(f"Generating {cfg.batch_size} Molecules!")
    model.eval()
    
    with torch.no_grad():
        sequence, adj_matrix, special_token = model.generate(
            num_data=cfg.batch_size, 
            num_steps=cfg.steps, 
            top_k=cfg.top_k
        )
        
        logger.info(f"Done generating '{cfg.data}' computing metrics!")
        
        if cfg.data == 'mols':
            valid_mols = []
            invalid_mols = []
            smiles_list = []
            
            for i in range(sequence.shape[0]):
                curr_num_nodes = (special_token[i] == 0).nonzero()[0,0].item()
                curr_sequence = sequence[i,:curr_num_nodes, 0].to(torch.int)
                curr_adj = adj_matrix[i,:curr_num_nodes, :curr_num_nodes]
                valid, smiles, mol = is_valid_molecule(curr_sequence, curr_adj, verbose=False)
                
                if valid:
                    valid_mols.append(mol)
                    smiles_list.append(smiles)
                else:
                    invalid_mols.append(mol) 
                    if cfg.save_sequence:
                        sub_dir = os.path.join(cfg.exp_name, 'invalid_pickle')
                        save_mol_pickle(curr_sequence, curr_adj, sub_dir=sub_dir, main_dir=cfg.mol_dir)
                    
            unique_mols = set(smiles_list)
            
            if cfg.draw:
                unique_mols_index = [index for index, value in enumerate(smiles_list) if value not in smiles_list[:index]]
                unique_mols_list =  [valid_mols[i] for i in unique_mols_index]
                assert len(unique_mols) == len(unique_mols_list)
                
                for mol in unique_mols_list:
                    sub_dir = os.path.join(cfg.exp_name, 'valid_image')
                    draw_mol(mol, sub_dir=sub_dir, main_dir=cfg.mol_dir)
                    
                for mol in invalid_mols:
                    sub_dir = os.path.join(cfg.exp_name, 'invalid_image')
                    draw_mol(mol, sub_dir=sub_dir, main_dir=cfg.mol_dir)
            
            logger.info(f"Validity: {len(smiles_list)/cfg.batch_size}")
            logger.info(f"Uniqueness: {len(unique_mols)/len(smiles_list)}, amount:{len(unique_mols)}, ratio:{len(unique_mols)/cfg.batch_size}")
            
        elif cfg.data == 'grid':
            folder_path = os.path.join(cfg.root, cfg.mol_dir, cfg.exp_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            existing_files = os.listdir(folder_path)
            existing_image_files = [f for f in existing_files if f.endswith('.png')]  # List only PNG files
            index = len(existing_image_files)
            
            for i in range(sequence.shape[0]):
                curr_num_nodes = (special_token[i] == 0).nonzero()[0,0].item()
                curr_sequence = sequence[i,:curr_num_nodes, 0].to(torch.int)
                curr_adj = adj_matrix[i,:curr_num_nodes, :curr_num_nodes]
                curr_adj.diagonal(dim1=0, dim2=1).fill_(0)
                
                # draw and save
                A_np = curr_adj.cpu().numpy()
                G = nx.from_numpy_array(A_np)
                plt.figure(figsize=(8, 6))
               
                pos = nx.spring_layout(G, seed=0)

                nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=200, font_size=12, font_weight='bold', edge_color='gray')
                plt.savefig(os.path.join(folder_path, f'{index}.png'))
                index+=1


if __name__ == "__main__":
    main()