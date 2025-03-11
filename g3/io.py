from dataclasses import dataclass
from typing import Tuple
from g3.gen_gpt import GenGPT
from g3.gen_gpt_edge import GenGPT_Edge
import torch
import numpy as np
import os
import wandb
from loguru import logger


class Loader:
    def __init__(self, root, dataset_name, data_shapes, batch_size):
        train_shape, val_shape = data_shapes
        file_path = f"{root}/data/{dataset_name}/train.bin"

        self.train_data = np.memmap(
            f"{root}/data/{dataset_name}/train.bin", dtype=np.uint16, mode="r", shape=train_shape
        )
        self.cond_train_data = torch.load(f"{root}/data/{dataset_name}/train.pkl")

        self.val_data = np.memmap(
            f"{root}/data/{dataset_name}/val.bin", dtype=np.uint16, mode="r", shape=val_shape
        )
        self.cond_val_data = torch.load(f"{root}/data/{dataset_name}/val.pkl")
        self.batch_size = batch_size

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        cond_data = self.cond_train_data if split == "train" else self.cond_val_data
        
        num_nodes_batch_list = cond_data['num_nodes_batch']
        special_target_list = cond_data['special_target']
        edge_type_target_list = cond_data['edge_type_target']
        attn_bias = cond_data['attn_bias']
        
        batch_idx = np.random.randint(0, len(data), (self.batch_size,), dtype='int')
        batch = torch.from_numpy(data[batch_idx].astype(np.int64))
        
        num_nodes_batch = []
        attn_bias_batch = []
        special_target_batch = []
        edge_type_target_batch = []
    
        for batch_id in batch_idx.tolist():
            num_nodes = num_nodes_batch_list[batch_id]  
            num_nodes_batch.append(num_nodes)
            special_target_batch.append(special_target_list[batch_id])
            edge_type_target_batch.append(edge_type_target_list[batch_id])
            attn_bias_batch.append(attn_bias[batch_id].unsqueeze(dim=0))
        
        merged_num_nodes = torch.tensor(num_nodes_batch)
        merged_attn_bias = torch.cat(attn_bias_batch, dim=0).to(torch.int)
        merged_special_target = torch.cat(special_target_batch, dim=0)
        merged_edge_type_target = torch.cat(edge_type_target_batch, dim=0)
        
        return batch, merged_num_nodes, merged_attn_bias ,merged_special_target, merged_edge_type_target 


def ensure_root_folder(root):
    if not os.path.exists(data_dir := f"{root}/data"):
        logger.info(f"Creating data directory {data_dir}")
        os.makedirs(data_dir)

    if not os.path.exists(ckpt_dir := f"{root}/ckpt"):
        logger.info(f"Creating ckpt directory {ckpt_dir}")
        os.makedirs(ckpt_dir)

    return data_dir, ckpt_dir


def load_inference_model(root, checkpoint_name, compile, device, edge_attn):
    checkpoint_path = get_checkpoint_path(root, checkpoint_name)
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint {checkpoint_name} not found, got {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if edge_attn:
        model = GenGPT_Edge(checkpoint["model_args"])
    else:
        model = GenGPT(checkpoint["model_args"])
        
    model.cfg.device = device
    load_model_checkpoint(model, checkpoint)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model


@dataclass(frozen=True)
class Metadata:
    block_size: int
    embed_map: dict
    data_shapes: Tuple[torch.Size, torch.Size]
  

def load_metadata(datadir):
    meta_data_path = f"{datadir}/metadata.pt"
    logger.info(f"Loading metadata at {meta_data_path}")
    meta_data = torch.load(meta_data_path)
    block_size = meta_data["max_num_nodes"]
    logger.info(f"Found block size {block_size}")

    embed_map = dict(
        node_type = meta_data["num_node_type"],
        special_type=meta_data["num_special_type"],
        edge_type= meta_data["num_edge_type"],
        degree_type = meta_data["num_degree_type"]
        
    )
    logger.info(f"Found embedding map {embed_map}")

    train_shape = meta_data["train_shape"]
    val_shape = meta_data["val_shape"]
    logger.info(f"Found training data of shape {train_shape}")
    logger.info(f"Found validation data of shape {val_shape}")

    return Metadata(block_size, embed_map, (train_shape, val_shape))


def get_checkpoint_path(root, checkpoint):
    return f"{root}/ckpt/{checkpoint}.pt"


def load_model_checkpoint(model, checkpoint):
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)


def load_checkpoint(root, checkpoint, model, optimizer, device):
    iter_num = 0
    best_ratio = -1

    if checkpoint is not None:
        checkpoint_path = get_checkpoint_path(root, checkpoint)
        logger.info(f"Trying to load checkpoint from {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            load_model_checkpoint(model, checkpoint)
            optimizer.load_state_dict(checkpoint["optimizer"])

            iter_num = checkpoint["iter_num"]
            best_ratio = checkpoint["best_ratio"]
            logger.info(f"Checkpoint found. Continuing at iteration {iter_num}")
        else:
            logger.info("Checkpoint not found. Training from scratch")
    else:
        logger.info("Training from scratch")

    return iter_num, best_ratio


def save_checkpoint(
    root, checkpoint_name, model, optimizer, model_args, iter_num, best_ratio
):
    if iter_num > 0 and checkpoint_name is not None:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "iter_num": iter_num,
            "best_ratio": best_ratio,
        }
        checkpoint_path = f"{root}/ckpt/{checkpoint_name}.pt"
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
    else:
        logger.info(
            "No checkpoint name provided. To save checkpoints, make sure to set checkpoint in the command line"
        )


def setup_wandb(cfg):
    if cfg.wandb_project is not None:
        wandb_dir = cfg.wandb_dir if cfg.wandb_dir is not None else cfg.root
        logger.info(f"Local wandb data saved to {wandb_dir}")
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            dir=wandb_dir,
            config=dict(cfg),
        )


def log_progress(iter_num, losses, gen_metrics, best_ratio, lr, running_mfu, wandb_project):
    
    metrics = {
        "generation/uniqueness": gen_metrics['uniqueness'],
        "generation/validity": gen_metrics['validity'],
        "generation/ratio": gen_metrics['ratio'],
        "best/ratio": best_ratio,
        **losses,
        "lr": lr,
        "mfu": running_mfu * 100,
    }
    metric_string = " | ".join([f"{key}: {val:.3f}" for key, val in metrics.items()])
    logger.info(f"step {iter_num}: {metric_string}")
    if wandb_project:
        wandb.log(metrics)
