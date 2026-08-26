#!/usr/bin/env python3
"""Fail-closed multi-node NCCL collective probe for the H200 IB path."""

import argparse
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def main():
    args = build_parser().parse_args()
    if args.expected_world_size < 2:
        raise ValueError("IB collective probe requires at least two ranks")
    if args.timeout_seconds < 1:
        raise ValueError("timeout must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NCCL collective probe")

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError("torchrun environment is missing " + ", ".join(missing))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected world size {args.expected_world_size}, got {world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    try:
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(
                f"rank {rank} all_reduce mismatch: {value.item()} != {expected_sum}"
            )

        record = {
            "hostname": socket.gethostname(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "cuda_device": torch.cuda.get_device_name(local_rank),
            "all_reduce_sum": value.item(),
        }
        records = [None] * world_size
        dist.all_gather_object(records, record)
        dist.barrier()
        if rank == 0:
            print(json.dumps({"status": "ok", "ranks": records}, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
