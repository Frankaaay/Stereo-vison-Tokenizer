"""Immutable training configuration and provenance records."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from stereo_tokenizer.mode_sampling import MODE_IDS, resolve_mode_int_spec


def _jsonable(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_immutable_json(path, payload):
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"refusing to overwrite mismatched run metadata {path}")
        return serialized
    path.write_text(serialized, encoding="utf-8")
    return serialized


def _is_global_zero_process(environ):
    """Identify rank zero before Lightning initializes the process group."""
    if "RANK" in environ:
        return int(environ["RANK"]) == 0
    return int(environ.get("NODE_RANK", "0")) == 0 and int(
        environ.get("LOCAL_RANK", "0")
    ) == 0


def write_online_gt_run_metadata(args):
    """Persist resolved backend provenance before an online-teacher run."""
    if not args.online_gt_enabled or not _is_global_zero_process(os.environ):
        return
    output_root = Path(args.default_root_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {key: _jsonable(value) for key, value in vars(args).items()}
    resolved_serialized = _write_immutable_json(
        output_root / "resolved_config.json", resolved
    )
    code_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    online_gt = {
        "backend": args.foundation_stereo_backend,
        "checkpoint_sha256": (
            args.las2_h_checkpoint_sha256
            if args.foundation_stereo_backend == "las2_h"
            else args.foundation_stereo_checkpoint_sha256
        ),
        "valid_iters": (
            args.las2_h_valid_iters
            if args.foundation_stereo_backend == "las2_h"
            else args.foundation_stereo_valid_iters
        ),
        "pair_microbatch": args.foundation_stereo_pair_microbatch,
        "bidirectional": True,
        "lr_consistency": True,
    }
    if args.foundation_stereo_backend == "las2_h":
        online_gt.update(
            {
                "repo": str(Path(args.las2_h_repo).resolve()),
                "source_sha": args.las2_h_source_sha,
                "checkpoint": str(Path(args.las2_h_checkpoint).resolve()),
                "max_disp": args.las2_h_max_disp,
            }
        )
    elif args.foundation_stereo_backend == "pytorch":
        online_gt.update(
            {
                "repo": str(Path(args.foundation_stereo_repo).resolve()),
                "checkpoint": str(
                    Path(args.foundation_stereo_checkpoint).resolve()
                ),
            }
        )
    else:
        online_gt.update(
            {
                "engine": str(Path(args.foundation_stereo_engine).resolve()),
                "engine_sha256": args.foundation_stereo_engine_sha256,
                "engine_manifest": str(
                    Path(args.foundation_stereo_engine_manifest).resolve()
                ),
                "engine_manifest_sha256": (
                    args.foundation_stereo_engine_manifest_sha256
                ),
            }
        )
    distributed = {
        "mode": args.distributed_mode,
        "num_nodes": args.num_nodes,
        "devices_per_node": args.devices,
        "expected_world_size": args.num_nodes * args.devices,
        "node_rank": os.environ.get("NODE_RANK"),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
    }
    run_manifest = {
        "schema": "stereo-vae-online-gt-run-v2",
        "code_sha": code_sha,
        "resolved_config_sha256": hashlib.sha256(
            resolved_serialized.encode("utf-8")
        ).hexdigest(),
        "distributed": distributed,
        "online_gt": online_gt,
    }
    if getattr(args, "continuation_checkpoint", None) is not None:
        run_manifest["continuation"] = {
            "checkpoint": str(args.continuation_checkpoint.resolve()),
            "checkpoint_sha256": args.continuation_checkpoint_sha256,
            "source_generator_updates": args.continuation_source_generator_updates,
            "source_contract": args.continuation_source_contract,
            "optimizer_restored": False,
            "scheduler_aligned_to_source_update": True,
        }
    mode_batch_sizes = resolve_mode_int_spec(
        args.mode_batch_sizes,
        fallback=int(args.batch_size),
    )
    mode_grad_accumulates = resolve_mode_int_spec(
        args.mode_grad_accumulates,
        fallback=int(args.grad_accumulates),
    )
    run_manifest["logical_update_batch_contract"] = {
        mode_id: {
            "per_device_batch_size": mode_batch_sizes[mode_id],
            "micro_batches_per_logical_update": mode_grad_accumulates[mode_id],
            "effective_global_batch_size": mode_batch_sizes[mode_id]
            * mode_grad_accumulates[mode_id]
            * args.devices
            * args.num_nodes,
        }
        for mode_id in MODE_IDS
    }
    run_manifest["online_gt"]["da3"] = {
        "repo": str(Path(args.da3_repo).resolve()),
        "source_sha": args.da3_source_sha,
        "checkpoint": str(Path(args.da3_checkpoint).resolve()),
        "checkpoint_sha256": args.da3_checkpoint_sha256,
        "process_res": args.da3_process_res,
        "process_res_method": args.da3_process_res_method,
        "confidence_mask_mode": args.da3_confidence_mask_mode,
    }
    _write_immutable_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps({"online_gt_provenance": online_gt}, sort_keys=True))
