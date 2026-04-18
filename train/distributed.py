from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_gpu_ids(gpu_ids: Optional[str]) -> Optional[List[int]]:
    if gpu_ids is None:
        return None
    parsed = [part.strip() for part in gpu_ids.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--gpu-ids was provided but no valid GPU ids were found")
    return [int(part) for part in parsed]


def init_distributed_context(args) -> Dict[str, Any]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = os.environ.get("LOCAL_RANK")
    local_rank = args.local_rank
    if env_local_rank is not None:
        local_rank = int(env_local_rank)
    elif local_rank < 0:
        local_rank = 0

    distributed = world_size > 1
    use_cuda = torch.cuda.is_available() and args.device != "cpu"

    if distributed:
        if not use_cuda:
            raise ValueError("Distributed training currently requires CUDA")
        if gpu_ids is None:
            if torch.cuda.device_count() < world_size:
                raise ValueError(
                    f"WORLD_SIZE={world_size} exceeds available CUDA devices ({torch.cuda.device_count()})"
                )
            gpu_ids = list(range(world_size))
        if len(gpu_ids) != world_size:
            raise ValueError(
                f"WORLD_SIZE={world_size} but --gpu-ids resolved to {len(gpu_ids)} GPUs: {gpu_ids}"
            )
        if local_rank < 0 or local_rank >= len(gpu_ids):
            raise ValueError(
                f"LOCAL_RANK={local_rank} is out of range for --gpu-ids {gpu_ids}"
            )

        device_index = gpu_ids[local_rank]
        torch.cuda.set_device(device_index)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", device_index)
    else:
        if gpu_ids is not None:
            if not use_cuda:
                raise ValueError("--gpu-ids requires CUDA")
            if len(gpu_ids) != 1:
                raise ValueError(
                    "Multiple --gpu-ids require torchrun, e.g. `torchrun --nproc_per_node=2 train/finetune.py --gpu-ids 0,1 ...`"
                )
            device = torch.device("cuda", gpu_ids[0])
            torch.cuda.set_device(device)
        else:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main_process": rank == 0,
        "device": device,
        "gpu_ids": gpu_ids,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_run_timestamp(run_timestamp: Optional[str], distributed: bool) -> str:
    if not distributed:
        if run_timestamp is None:
            raise ValueError("run_timestamp must be provided when not distributed")
        return run_timestamp

    payload = [run_timestamp]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Failed to broadcast run timestamp from rank 0")
    return payload[0]
