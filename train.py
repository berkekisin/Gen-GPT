import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import hydra
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from loguru import logger
from g3.io import (
    Loader,
    load_metadata,
    load_checkpoint,
    save_checkpoint,
    setup_wandb,
    log_progress,
)
from g3.device import device_setup, to_device
from g3.utils import CosineWithWarmupLR, gradient_accumulation_setup, MultiTargetLoss
from g3.gen_gpt import  GenGPT
from g3.gen_gpt_edge import GenGPT_Edge
from g3.transformer_layers import GPTConfig
from g3.molecules import is_valid_molecule
 

@hydra.main(version_base=None, config_path="./config/", config_name="train")
def main(cfg):
    device_info = device_setup(cfg.root, cfg.backend, d=cfg.device)
    
    if cfg.casual_name:
        cfg.checkpoint_name = f"{cfg.dataset_name}-d{cfg.embed_dim}-b{cfg.bias}-nlayer:{cfg.num_layers}"
        cfg.wandb_name = cfg.checkpoint_name
        
    if device_info.master_process:
        setup_wandb(cfg)

    metadata = load_metadata(datadir=f"{cfg.root}/data/{cfg.dataset_name}")

    grad_accum_steps = gradient_accumulation_setup(
        cfg.grad_accum_steps, device_info.ddp_world_size
    )
    logger.info(f"Using {grad_accum_steps} grad. accum. steps on device {device_info.ddp_local_rank}")

    tokens_per_iter = (
        grad_accum_steps
        * device_info.ddp_world_size
        * cfg.batch_size
        * metadata.block_size
    )
    logger.info(f"Calculated tokens per iteration as {tokens_per_iter}")

    logger.info("Creating the model")
    gptconf = GPTConfig(
        edge_attn = cfg.edge_attn,
        embed_map=metadata.embed_map,
        target_sizes = cfg.target_sizes,
        block_size=metadata.block_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        bias=cfg.bias,
        device=cfg.device
    )
    if cfg.edge_attn:
        model = GenGPT_Edge(gptconf).to(device_info.device)
        logger.info("Using Edge Attention!")
    else:
        model = GenGPT(gptconf).to(device_info.device)
        logger.info("Using Normal Attention!")
        
    logger.info("Number of parameters: %.2fM" % (model.get_num_params() / 1e6,))

    logger.info("Creating loader, scaler, optimizer and scheduler")
    loader = Loader(
        cfg.root, cfg.dataset_name, metadata.data_shapes, cfg.batch_size
    )
    scaler = torch.cuda.amp.GradScaler('cuda', enabled=(device_info.dtype == "float16"))
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (0.9, 0.95), device_info.device_type
    )
    scheduler = CosineWithWarmupLR(
        optimizer,
        cfg.warmup_iters,
        cfg.learning_rate,
        cfg.num_iters,
        cfg.learning_rate // 10,
    )
    
    iter_num, best_ratio = load_checkpoint(
        cfg.root, cfg.checkpoint_name, model, optimizer, device_info.device
    )

    if cfg.compile:
        logger.info("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    #if device_info.ddp_world_size > 1:
    #    model = DDP(model, device_ids=[device_info.ddp_local_rank])

    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if device_info.ddp_world_size > 1 else model
    running_mfu = -1.0

    # loss settings
    loss_keys = ['node_type', 'edge_type', 'special_type']
    loss_coeffs = {key: cfg.loss_coeff for key in loss_keys}

    @torch.no_grad()
    def estimate_loss(split):
        loss = MultiTargetLoss(loss_keys )
        node_type_acc = 0
        edge_type_acc = 0
        special_type_acc = 0
        f1_edge = 0
        
        for _ in range(cfg.eval_iters):
            data = to_device(
                loader.get_batch(split), device_info.device_type, device_info.device
            )
            with device_info.ctx:
                loss_returned, accs, macro_f1_edge = model(*data)
                loss += loss_returned
                node_type_acc += accs['node_type']
                edge_type_acc += accs['edge_type']
                special_type_acc += accs['special_type']
                f1_edge += macro_f1_edge
                
        avg_loss, out_loss = loss.mean(loss_coeffs)

        out_loss = {f"{split}/{key}_loss": val for key, val in out_loss.items()}
        out_loss[f"{split}/loss"] = avg_loss
        out_loss[f'{split}/acc_node_type'] = node_type_acc/cfg.eval_iters
        out_loss[f'{split}/acc_edge_type'] = edge_type_acc/cfg.eval_iters
        out_loss[f'{split}/acc_special_type'] = special_type_acc/cfg.eval_iters
        out_loss[f'{split}/f1'] = f1_edge/cfg.eval_iters
        
        return out_loss
    
    logger.info('Starting Training!!')
    model.train()
    while True:
        lr = scheduler(iter_num)
        
        # Validation
        if (iter_num + 1) % cfg.eval_interval == 0 and device_info.master_process:
            logger.info("Validating the Model!")
            model.eval()
            losses= {**estimate_loss("train"), **estimate_loss("val")}
            
            # generate mols
            with torch.no_grad():
                
                gen_batch_size = cfg.gen.batch_size

                sequence, adj_matrix, special_token = model.generate(
                    num_data=gen_batch_size,
                    num_steps=cfg.gen.steps,
                    top_k=cfg.gen.top_k
                )
    
                valid_mols = []
                smiles_list = []
        
                for i in range(sequence.shape[0]):
                    curr_num_nodes = (special_token[i] == 0).nonzero()[0,0].item()
                    curr_sequence = sequence[i,:curr_num_nodes, 0].to(torch.int)
                    curr_adj = adj_matrix[i,:curr_num_nodes, :curr_num_nodes]
                    valid, smiles, mol = is_valid_molecule(curr_sequence, curr_adj, verbose=False)
            
                    if valid:
                        valid_mols.append(mol)
                        smiles_list.append(smiles)
                
                unique_mols = set(smiles_list)
                validity = len(smiles_list)/gen_batch_size
                uniqueness = 0 if len(smiles_list) == 0 else len(unique_mols)/len(smiles_list)
                ratio = len(unique_mols)/ gen_batch_size
                gen_metrics = {"validity": validity, "uniqueness": uniqueness, 'ratio': ratio}
                
            # save checkpoint
            if ratio > best_ratio:
                best_ratio = ratio
                save_checkpoint(cfg.root, cfg.checkpoint_name, raw_model, optimizer, gptconf, iter_num, best_ratio) 
           
            # log to wandb
            log_progress(iter_num, losses, gen_metrics, best_ratio, lr, running_mfu, cfg.wandb_project)
            logger.info("Validation Finished!")

        model.train()

        for micro_step in range(grad_accum_steps):
            if device_info.ddp_world_size > 1:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            data = to_device(
                loader.get_batch("train"), device_info.device_type, device_info.device
            )
            with device_info.ctx:
                loss_returned , _, _ = model(*data)
                loss = MultiTargetLoss(loss_returned).sum(loss_coeffs)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        #logger.info(f"time needed: {dt}")
        
        if iter_num % cfg.log_interval == 0 and device_info.master_process:
            lossf = loss.item() * grad_accum_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(cfg.batch_size * grad_accum_steps, dt)
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            logger.info(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.num_iters:
            break

    logger.info("Training complete ✨")
    if device_info.ddp_world_size > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
