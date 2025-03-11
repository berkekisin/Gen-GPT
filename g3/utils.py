import math
import os
import pandas as pd
from omegaconf import ListConfig

import torch
import torch.nn.functional as F
from torch_geometric.utils import cumsum


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, indices = torch.topk(logits, top_k, dim=-1)

        # Step 2: Create a mask for the top-k elements
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))

        # Step 3: Filter the tensor, keeping only top-k elements
        logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))

        #v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        #logits[logits < v[:, [-1]]] = filter_value

    if top_p > 0.0:
        # NOTE: Based on Hugginface implementation (don't want to make transformers library a dependency
        # just for this but eventually we should move to transformers library.)
        # https://github.com/huggingface/transformers/blob/e547458c43dfdbbb8f6a7757237e234c44e20a8f/src/transformers/generation/logits_process.py#L447
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def pad_2d_tensor(tensor, pad_amount, pad_value=0):
    """
    Pads a 2D tensor on the right and bottom sides with a given value.
    Args:
        tensor (torch.Tensor): The input 2D tensor of shape (H, W).
        pad_right (int): Padding size for the right side.
        pad_bottom (int): Padding size for the bottom side.
        pad_value (int, float): The value to pad with.

    Returns:
        torch.Tensor: The padded 2D tensor.
    """
    pad = (0, pad_amount, 0, pad_amount)  # Left, Right, Top, Bottom
    return F.pad(tensor, pad, mode='constant', value=pad_value)
    

def get_dense_mask(x, num_nodes_batch):
    max_num_nodes = x.shape[1]
    batch_size = x.shape[0]
    cum_nodes = cumsum(num_nodes_batch)
    batch = torch.repeat_interleave(num_nodes_batch)
    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)
    
    return mask


class CosineWithWarmupLR:
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def __call__(self, epoch: int):
        lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self, epoch: int):
        # 1) linear warmup for warmup_iters steps
        if epoch < self.warmup_iters:
            return self.lr * epoch / self.warmup_iters
        # 2) if epoch > lr_decay_iters, return min learning rate
        if epoch > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)


def gradient_accumulation_setup(grad_accum_steps, ddp_world_size):
    assert (
        grad_accum_steps % ddp_world_size == 0
    ), f"Gradient accumulation steps must be divisible by world size, got {grad_accum_steps} and {ddp_world_size}"
    grad_accum_steps //= ddp_world_size
    return grad_accum_steps


class MultiTargetLoss:
    def __init__(self, mt_input, num_updates=1):
        if isinstance(mt_input, list):
            self.mt_loss = {key: 0.0 for key in mt_input}
            self.num_updates = 0
        elif isinstance(mt_input, dict):
            self.mt_loss = mt_input
            self.num_updates = num_updates
        else:
            raise ValueError(
                f"Input must either be a list of keys or an existing loss dict"
            )

    def __add__(self, other):
        return MultiTargetLoss(self.__add_loss(other), self.num_updates + 1)

    def __iadd__(self, other):
        self.mt_loss = self.__add_loss(other)
        self.num_updates += 1
        return self

    def __add_loss(self, other):
        assert isinstance(
            other, dict
        ), f"Operation not supported for type {type(other)}"
        return {
            key: self.mt_loss[key] + other[key].item() for key in self.mt_loss.keys()
        }

    def sum(self, loss_coeffs):
        total_loss = 0.0
        for key, loss in self.mt_loss.items():
            total_loss += loss_coeffs[key] * loss
        return total_loss

    def mean(self, loss_coeffs):
        if self.num_updates == 0:
            raise ValueError("Cannot compute mean from empty MultiTargetLoss")
        total_loss = self.sum(loss_coeffs)
        avg_loss = total_loss / self.num_updates
        avg_mt_loss = {
            key: value / self.num_updates for key, value in self.mt_loss.items()
        }
        return avg_loss, avg_mt_loss


def listify(obj):
    if not isinstance(obj, (list, ListConfig)):
        return [obj]
    return obj


def save_metrics(cfg, results_file, metrics):
    results_file_exists = os.path.exists(results_file)
    results = pd.DataFrame([{**dict(cfg), **metrics}])
    results.to_csv(
        results_file,
        index=False,
        header=not results_file_exists,
        mode="a" if results_file_exists else "w",
    )
