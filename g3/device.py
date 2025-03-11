import torch
import os
from torch.distributed import init_process_group
from contextlib import nullcontext, AbstractContextManager
from loguru import logger
from dataclasses import dataclass
from g3.io import ensure_root_folder


PTDTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class DeviceInfo:
    master_process: bool
    ddp_world_size: int
    ddp_local_rank: int
    ctx: AbstractContextManager
    device_type: str
    device: torch.device
    dtype: torch.dtype


def device_setup(root, backend=None, d='cuda'):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = d
    logger.info(f"Selecting device: {device}")
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    logger.info(f"Selecting dtype: {dtype}")

    is_ddp = backend is not None and int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        logger.info("Detected DDP run")
        init_process_group(backend=backend)

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        logger.info("Running on a single CPU/GPU")
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0

    if master_process:
        ensure_root_folder(root)

    #torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"

    ptdtype = PTDTYPE[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    return DeviceInfo(
        master_process, ddp_world_size, ddp_local_rank, ctx, device_type, device, dtype
    )


def to_device(tensors, device_type, device):
    out_tensors = []
    for tensor in tensors:
        if tensor is None:
            out_tensors.append(tensor)
        elif device_type == "cuda":
            out_tensors.append(tensor.pin_memory().to(device, non_blocking=True))
        else:
            out_tensors.append(tensor.to(device))
    return out_tensors
