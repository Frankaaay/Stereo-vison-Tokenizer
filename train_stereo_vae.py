import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.profiler import ProfilerActivity

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    mode_occurrences_before,
    parse_weight_spec,
    resolve_mode_int_spec,
)
from stereo_tokenizer.modules.callbacks import ImageLogger, VideoLogger
from stereo_tokenizer.online_gt import (
    OnlineDepthAnything3GTCallback,
    OnlineFoundationGTCallback,
    validate_tensorrt_engine_assets,
)
from stereo_tokenizer.profiling import set_profiling_enabled


def _timing_summary(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


class StepTimingCallback(Callback):
    def __init__(self, output_path, warmup_updates):
        self.output_path = Path(output_path)
        self.warmup_updates = warmup_updates
        self.last_batch_end = None
        self.current_batch_start = None
        self.current_input_wait = None
        self.logical_update_start = None
        self.last_generator_update = None
        self.timings = []
        self.micro_timings = []

    def on_train_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.last_batch_end = time.perf_counter()
        self.last_generator_update = int(pl_module.generator_updates)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.current_input_wait = now - self.last_batch_end
        self.current_batch_start = now
        if int(pl_module._micro_step) == 0:
            self.logical_update_start = self.last_batch_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        now = time.perf_counter()
        micro_interval = now - self.current_batch_start
        generator_update = int(pl_module.generator_updates)
        logical_step = generator_update
        if int(pl_module.last_micro_step_index) < int(
            pl_module.last_accumulation_factor
        ):
            logical_step += 1
        micro_global_samples = int(pl_module.last_microbatch_size) * int(
            trainer.world_size
        )
        self.micro_timings.append(
            {
                "batch_update": int(pl_module.batch_updates),
                "logical_step": logical_step,
                "mode_id": pl_module.last_mode_id,
                "temporal_mode": pl_module.last_temporal_mode,
                "micro_step_index": int(pl_module.last_micro_step_index),
                "accumulation_factor": int(pl_module.last_accumulation_factor),
                "per_device_batch_size": int(pl_module.last_microbatch_size),
                "global_samples": micro_global_samples,
                "samples_per_s": micro_global_samples / micro_interval,
                "input_wait_and_transfer_s": self.current_input_wait,
                "interval_s": micro_interval,
            }
        )
        self.last_batch_end = now
        if generator_update == self.last_generator_update:
            return
        if generator_update != self.last_generator_update + 1:
            raise RuntimeError("step timing observed a non-consecutive generator update")
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        logical_interval = now - self.logical_update_start
        global_samples = int(pl_module.last_logical_global_samples)
        self.timings.append(
            {
                "step": generator_update,
                "mode_id": pl_module.last_mode_id,
                "temporal_mode": pl_module.last_temporal_mode,
                "interval_s": logical_interval,
                "global_samples": global_samples,
                "samples_per_s": global_samples / logical_interval,
                "peak_memory_allocated_bytes": peak_allocated,
                "peak_memory_reserved_bytes": peak_reserved,
            }
        )
        mode_prefix = f"train/{pl_module.last_mode_id}"
        pl_module.log_dict(
            {
                f"{mode_prefix}/step_time_s": logical_interval,
                f"{mode_prefix}/samples_per_s": global_samples / logical_interval,
                f"{mode_prefix}/peak_memory_allocated_bytes": float(
                    peak_allocated
                ),
                f"{mode_prefix}/peak_memory_reserved_bytes": float(
                    peak_reserved
                ),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        torch.cuda.reset_peak_memory_stats()
        self.last_generator_update = generator_update
        self.logical_update_start = None

    def on_train_end(self, trainer, pl_module):
        local_memory = torch.tensor(
            [
                [
                    max(
                        (
                            row["peak_memory_allocated_bytes"]
                            for row in self.timings
                            if row["mode_id"] == mode_id
                        ),
                        default=0,
                    ),
                    max(
                        (
                            row["peak_memory_reserved_bytes"]
                            for row in self.timings
                            if row["mode_id"] == mode_id
                        ),
                        default=0,
                    ),
                ]
                for mode_id in MODE_IDS
            ],
            device=pl_module.device,
            dtype=torch.long,
        )
        rank_memory = [local_memory]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_memory = [
                torch.zeros_like(local_memory) for _ in range(trainer.world_size)
            ]
            torch.distributed.all_gather(rank_memory, local_memory)
        if not trainer.is_global_zero:
            return
        stable = []
        mode_seen = {mode_id: 0 for mode_id in MODE_IDS}
        for row in self.timings:
            mode_id = row["mode_id"]
            if mode_seen[mode_id] >= self.warmup_updates:
                stable.append(row)
            mode_seen[mode_id] += 1
        values = [row["interval_s"] for row in stable]
        stable_by_temporal_mode = {}
        for temporal_mode in ("single_frame", "four_frame"):
            mode_values = [
                row["interval_s"]
                for row in stable
                if row["temporal_mode"] == temporal_mode
            ]
            if mode_values:
                stable_by_temporal_mode[temporal_mode] = _timing_summary(
                    mode_values
                )
        stable_by_mode = {}
        stable_samples_per_s_by_mode = {}
        for mode_id in MODE_IDS:
            mode_values = [
                row["interval_s"]
                for row in stable
                if row["mode_id"] == mode_id
            ]
            if mode_values:
                stable_by_mode[mode_id] = _timing_summary(mode_values)
                stable_samples_per_s_by_mode[mode_id] = _timing_summary(
                    [
                        row["samples_per_s"]
                        for row in stable
                        if row["mode_id"] == mode_id
                    ]
                )
        payload = {
            "world_size": int(trainer.world_size),
            "per_mode_batch_sizes": dict(pl_module.mode_batch_sizes),
            "per_mode_grad_accumulates": dict(pl_module.mode_grad_accumulates),
            "per_mode_effective_global_batch_sizes": {
                mode_id: pl_module.mode_batch_sizes[mode_id]
                * pl_module.mode_grad_accumulates[mode_id]
                * int(trainer.world_size)
                for mode_id in MODE_IDS
            },
            "warmup_updates_per_mode": self.warmup_updates,
            "peak_memory_bytes_by_rank_and_mode": [
                {
                    "rank": rank,
                    "modes": {
                        mode_id: {
                            "allocated": int(memory[index, 0].item()),
                            "reserved": int(memory[index, 1].item()),
                        }
                        for index, mode_id in enumerate(MODE_IDS)
                    },
                }
                for rank, memory in enumerate(rank_memory)
            ],
            "timings": self.timings,
            "micro_timings": self.micro_timings,
            "stable": _timing_summary(values),
            "stable_by_temporal_mode": stable_by_temporal_mode,
            "stable_by_mode": stable_by_mode,
            "stable_samples_per_s_by_mode": stable_samples_per_s_by_mode,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class DiscriminatorExpansionOptimizerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        source_states = getattr(
            pl_module, "_discriminator_expansion_optimizer_states", None
        )
        if not isinstance(source_states, list) or len(source_states) != 2:
            raise ValueError(
                "discriminator expansion requires exactly two source optimizer states"
            )
        optimizers = list(trainer.optimizers)
        if len(optimizers) != 2:
            raise ValueError(
                "discriminator expansion requires generator and discriminator optimizers"
            )

        generator_state = copy.deepcopy(source_states[0])
        generator_target = optimizers[0].state_dict()
        self._validate_group_shape(
            generator_state, generator_target, label="generator", exact=True
        )
        optimizers[0].load_state_dict(generator_state)

        image_param_count = len(list(pl_module.image_discriminator.parameters()))
        video_param_count = len(list(pl_module.video_discriminator.parameters()))
        discriminator_state = copy.deepcopy(source_states[1])
        discriminator_target = optimizers[1].state_dict()
        self._validate_group_shape(
            discriminator_state,
            discriminator_target,
            label="discriminator",
            exact=False,
        )
        source_group = discriminator_state["param_groups"][0]
        target_group = discriminator_target["param_groups"][0]
        source_ids = list(source_group["params"])
        target_ids = list(target_group["params"])
        if len(source_ids) != image_param_count:
            raise ValueError(
                "source discriminator optimizer does not match image discriminator"
            )
        if len(target_ids) != image_param_count + video_param_count:
            raise ValueError(
                "target discriminator optimizer does not match image+video discriminators"
            )
        source_state = discriminator_state.get("state", {})
        unknown_state_ids = set(source_state) - set(source_ids)
        if unknown_state_ids:
            raise ValueError(
                "source discriminator optimizer has state outside its parameter group"
            )
        discriminator_state["state"] = {
            target_ids[index]: source_state[source_id]
            for index, source_id in enumerate(source_ids)
            if source_id in source_state
        }
        source_group["params"] = target_ids
        optimizers[1].load_state_dict(discriminator_state)

        schedulers = pl_module._as_sequence(pl_module.lr_schedulers())
        if len(schedulers) != 2:
            raise ValueError(
                "discriminator expansion requires generator and discriminator schedulers"
            )
        schedulers[0].step_update(pl_module.generator_updates)
        schedulers[1].step_update(pl_module.discriminator_updates)
        del pl_module._discriminator_expansion_optimizer_states
        pl_module.discriminator_expansion_optimizer_restored = True

    @staticmethod
    def _validate_group_shape(source, target, *, label, exact):
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise TypeError(f"{label} optimizer state must be a mapping")
        source_groups = source.get("param_groups")
        target_groups = target.get("param_groups")
        if not isinstance(source_groups, list) or len(source_groups) != 1:
            raise ValueError(f"{label} source optimizer must have one parameter group")
        if not isinstance(target_groups, list) or len(target_groups) != 1:
            raise ValueError(f"{label} target optimizer must have one parameter group")
        if exact and len(source_groups[0]["params"]) != len(
            target_groups[0]["params"]
        ):
            raise ValueError(f"{label} optimizer parameter count changed")


def _profile_event_time_us(event, name):
    value = getattr(event, name, None)
    if value is not None:
        return float(value)
    legacy = {
        "device_time_total": "cuda_time_total",
        "self_device_time_total": "self_cuda_time_total",
    }.get(name)
    return float(getattr(event, legacy, 0.0)) if legacy else 0.0


class TrainingTraceWriter:
    def __init__(self, output_dir, configured_active_steps):
        self.output_dir = Path(output_dir)
        self.configured_active_steps = configured_active_steps
        self.trace_index = 0

    @staticmethod
    def _row(event, denominator):
        return {
            "name": event.key,
            "calls": int(event.count),
            "cpu_total_ms_per_step": (
                float(event.cpu_time_total) / 1000.0 / denominator
            ),
            "cpu_self_ms_per_step": (
                float(event.self_cpu_time_total) / 1000.0 / denominator
            ),
            "device_total_ms_per_step": (
                _profile_event_time_us(event, "device_time_total")
                / 1000.0
                / denominator
            ),
            "device_self_ms_per_step": (
                _profile_event_time_us(event, "self_device_time_total")
                / 1000.0
                / denominator
            ),
        }

    def __call__(self, profiler):
        averages = list(profiler.key_averages())
        training_step = next(
            (
                event
                for event in averages
                if event.key == "stereo/step/training_step"
            ),
            None,
        )
        denominator = (
            int(training_step.count)
            if training_step is not None and training_step.count > 0
            else self.configured_active_steps
        )
        rows = [self._row(event, denominator) for event in averages]
        regions = [
            row
            for row in rows
            if row["name"].startswith("stereo/")
            or "DataLoader" in row["name"]
        ]
        regions.sort(
            key=lambda row: (
                row["device_total_ms_per_step"],
                row["cpu_total_ms_per_step"],
            ),
            reverse=True,
        )
        rows.sort(key=lambda row: row["device_self_ms_per_step"], reverse=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"rank0-trace-{self.trace_index:02d}"
        for filename, payload in (
            (
                f"{suffix}-regions.json",
                {"observed_active_steps": denominator, "regions": regions},
            ),
            (
                f"{suffix}-top-operators.json",
                {"observed_active_steps": denominator, "operators": rows[:100]},
            ),
        ):
            (self.output_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        profiler.export_chrome_trace(
            str(self.output_dir / f"{suffix}.json.gz")
        )
        self.trace_index += 1


class TrainingProfilerStepCallback(Callback):
    def __init__(self, profiler):
        self.profiler = profiler
        self.last_generator_update = None

    def on_train_start(self, trainer, pl_module):
        self.last_generator_update = int(pl_module.generator_updates)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        generator_update = int(pl_module.generator_updates)
        if generator_update == self.last_generator_update:
            return
        if generator_update != self.last_generator_update + 1:
            raise RuntimeError("profiler observed a non-consecutive generator update")
        self.last_generator_update = generator_update
        self.profiler.step()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--disable_media_logging", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="stereo-vae")
    parser.add_argument("--image_log_every_n_steps", type=int, default=750)
    parser.add_argument("--video_log_every_n_steps", type=int, default=1500)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--distributed_mode", choices=("single", "ib"), default="single"
    )
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--continuation_checkpoint", type=Path, default=None)
    parser.add_argument("--stage_transition_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--discriminator_expansion_checkpoint", type=Path, default=None
    )
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500)
    parser.add_argument("--step_timing_output", type=str, default=None)
    parser.add_argument("--step_timing_warmup", type=int, default=5)
    parser.add_argument("--torch_profile_output_dir", type=Path, default=None)
    parser.add_argument("--torch_profile_wait", type=int, default=5)
    parser.add_argument("--torch_profile_warmup", type=int, default=2)
    parser.add_argument("--torch_profile_active", type=int, default=4)
    parser.add_argument("--online_gt_enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--foundation_stereo_backend",
        choices=("las2_h", "pytorch", "tensorrt"),
        default="las2_h",
    )
    parser.add_argument("--foundation_stereo_repo", type=str, default=None)
    parser.add_argument("--foundation_stereo_checkpoint", type=str, default=None)
    parser.add_argument(
        "--foundation_stereo_checkpoint_sha256", type=str, default=None
    )
    parser.add_argument(
        "--foundation_stereo_valid_iters",
        type=int,
        choices=(12, 16, 32),
        default=16,
    )
    parser.add_argument(
        "--foundation_stereo_pair_microbatch", type=int, default=48
    )
    parser.add_argument("--foundation_stereo_engine", type=str, default=None)
    parser.add_argument(
        "--foundation_stereo_engine_sha256", type=str, default=None
    )
    parser.add_argument(
        "--foundation_stereo_engine_manifest", type=str, default=None
    )
    parser.add_argument(
        "--foundation_stereo_engine_manifest_sha256", type=str, default=None
    )
    parser.add_argument("--las2_h_repo", type=str, default=None)
    parser.add_argument("--las2_h_source_sha", type=str, default=None)
    parser.add_argument("--las2_h_checkpoint", type=str, default=None)
    parser.add_argument("--las2_h_checkpoint_sha256", type=str, default=None)
    parser.add_argument("--las2_h_valid_iters", type=int, default=4)
    parser.add_argument("--las2_h_max_disp", type=int, default=192)
    parser.add_argument(
        "--online_gt_cache_enabled", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--online_gt_cache_root", type=str, default=None)
    parser.add_argument(
        "--online_val_check_interval_steps", type=int, default=500
    )
    parser.add_argument("--da3_repo", type=str, default=None)
    parser.add_argument("--da3_source_sha", type=str, default=None)
    parser.add_argument("--da3_checkpoint", type=str, default=None)
    parser.add_argument("--da3_checkpoint_sha256", type=str, default=None)
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument(
        "--da3_process_res_method",
        choices=("upper_bound_resize",),
        default="upper_bound_resize",
    )
    parser.add_argument(
        "--da3_confidence_mask_mode",
        choices=("finite_positive_non_padding",),
        default="finite_positive_non_padding",
    )
    parser = StereoVAE.add_model_specific_args(parser)
    parser = StereoDataModule.add_data_specific_args(parser)
    return parser


def validate_distributed_runtime_args(args, environ=None):
    environ = os.environ if environ is None else environ
    expected_world_size = args.devices * args.num_nodes
    if args.distributed_mode == "single":
        if args.num_nodes != 1:
            raise ValueError("single distributed mode requires num_nodes=1")
    elif args.distributed_mode == "ib":
        if args.num_nodes != 2:
            raise ValueError("ib distributed mode requires num_nodes=2")
        missing = [
            name
            for name in ("NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE")
            if not environ.get(name)
        ]
        if missing:
            raise ValueError(
                "ib distributed mode requires environment variables "
                + ", ".join(missing)
            )
        try:
            node_rank = int(environ["NODE_RANK"])
            launcher_world_size = int(environ["WORLD_SIZE"])
            local_world_size = int(environ.get("LOCAL_WORLD_SIZE", args.devices))
        except ValueError as error:
            raise ValueError("distributed rank and world sizes must be integers") from error
        if not 0 <= node_rank < args.num_nodes:
            raise ValueError("NODE_RANK is outside the configured node range")
        if launcher_world_size != expected_world_size:
            raise ValueError(
                "torchrun WORLD_SIZE does not match devices times num_nodes"
            )
        if local_world_size != args.devices:
            raise ValueError("torchrun LOCAL_WORLD_SIZE does not match devices")
    else:
        raise ValueError(f"unsupported distributed mode {args.distributed_mode}")

    if environ.get("WORLD_SIZE"):
        try:
            launcher_world_size = int(environ["WORLD_SIZE"])
        except ValueError as error:
            raise ValueError("WORLD_SIZE must be an integer") from error
        if launcher_world_size != expected_world_size:
            raise ValueError(
                "launcher WORLD_SIZE does not match devices times num_nodes"
            )


def _validate_four_mode_batch_contract(args):
    if args.grad_accumulates != 1:
        raise ValueError(
            "four-mode training keeps global grad_accumulates=1 and uses "
            "mode_grad_accumulates"
        )
    if args.batch_size < 1:
        raise ValueError("four-mode batch size must be positive")
    world_size = args.devices * args.num_nodes
    if world_size < 1:
        raise ValueError("four-mode DDP world size must be positive")
    mode_batch_sizes = resolve_mode_int_spec(
        getattr(args, "mode_batch_sizes", None),
        fallback=int(args.batch_size),
    )
    mode_grad_accumulates = resolve_mode_int_spec(
        getattr(args, "mode_grad_accumulates", None),
        fallback=int(args.grad_accumulates),
    )
    if any(
        mode_grad_accumulates[mode_id] != 1
        for mode_id in MODE_IDS
        if mode_id.startswith("mono/")
    ):
        raise ValueError("mono modes currently require accumulation factor 1")
    effective_global_batches = {
        mode_id: mode_batch_sizes[mode_id]
        * mode_grad_accumulates[mode_id]
        * world_size
        for mode_id in MODE_IDS
    }
    if len(set(effective_global_batches.values())) != 1:
        raise ValueError(
            "four-mode effective global batch sizes must be equal: "
            f"{effective_global_batches}"
        )
    parse_weight_spec(args.mode_update_weights, MODE_IDS)
    parse_weight_spec(args.mono_dataset_weights, ("hy", "libero"))
    if args.num_nodes > 1 and not args.node_manifest_contracts:
        raise ValueError(
            "multi-node pretraining requires --node_manifest_contracts with "
            "the fixed node-rank to manifest mapping"
        )


def _resolve_val_check_interval(args):
    """Translate logical validation cadence to Lightning physical batches."""

    logical_interval = int(args.online_val_check_interval_steps)
    if not args.four_mode_mixed_training:
        return logical_interval

    remaining_updates = int(args.max_steps) - int(args.mode_schedule_start_update)
    logical_interval = min(logical_interval, remaining_updates)
    mode_weights = parse_weight_spec(args.mode_update_weights, MODE_IDS)
    cycle_size = sum(mode_weights.values())
    schedule_start = int(args.mode_schedule_start_update)
    periodic_interval = logical_interval < remaining_updates
    if periodic_interval and (
        logical_interval % cycle_size != 0 or schedule_start % cycle_size != 0
    ):
        raise ValueError(
            "four-mode periodic validation requires a cycle-aligned schedule start "
            f"and whole mode-schedule cycles ({cycle_size} logical updates), or "
            "the interval must cover all remaining updates"
        )
    accumulation = resolve_mode_int_spec(
        args.mode_grad_accumulates,
        fallback=int(args.grad_accumulates),
    )
    before = mode_occurrences_before(
        int(args.mode_schedule_seed),
        schedule_start,
        mode_weights,
    )
    after = mode_occurrences_before(
        int(args.mode_schedule_seed),
        schedule_start + logical_interval,
        mode_weights,
    )
    return sum(
        (after[mode_id] - before[mode_id]) * accumulation[mode_id]
        for mode_id in MODE_IDS
    )


def _bind_node_manifest_contracts(args):
    """Bind each physical node rank to immutable manifest content hashes."""
    if not args.four_mode_mixed_training:
        return
    local = {}
    for dataset_id in ("hy", "libero", "umi"):
        manifest = getattr(args, f"{dataset_id}_manifest")
        if not manifest:
            raise ValueError(f"three-source training requires {dataset_id}_manifest")
        path = Path(manifest).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        local[dataset_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    node_rank = str(int(os.environ.get("NODE_RANK", "0")))
    if args.node_manifest_contracts:
        candidate = Path(args.node_manifest_contracts).expanduser()
        raw = (
            candidate.read_text(encoding="utf-8")
            if not args.node_manifest_contracts.lstrip().startswith("{")
            and candidate.is_file()
            else args.node_manifest_contracts
        )
        try:
            contracts = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("node_manifest_contracts must be valid JSON") from error
    else:
        contracts = {node_rank: local}
    expected_ranks = {str(rank) for rank in range(int(args.num_nodes))}
    if not isinstance(contracts, dict) or set(contracts) != expected_ranks:
        raise ValueError(
            f"node manifest contracts must contain exactly ranks {sorted(expected_ranks)}"
        )
    for rank, mapping in contracts.items():
        if not isinstance(mapping, dict) or set(mapping) != {"hy", "libero", "umi"}:
            raise ValueError(f"node {rank} must map hy, libero and umi manifests")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            for digest in mapping.values()
        ):
            raise ValueError(f"node {rank} manifest hashes must be full SHA256 digests")
    if contracts[node_rank] != local:
        raise ValueError(
            f"node rank {node_rank} manifest files disagree with NODE_MANIFEST_CONTRACTS"
        )
    args.node_manifest_contracts = json.dumps(
        contracts, sort_keys=True, separators=(",", ":")
    )


def validate_runtime_args(args):
    checkpoint_args = (
        getattr(args, "resume_from_checkpoint", None),
        getattr(args, "continuation_checkpoint", None),
        getattr(args, "stage_transition_checkpoint", None),
        getattr(args, "discriminator_expansion_checkpoint", None),
    )
    if sum(value is not None for value in checkpoint_args) > 1:
        raise ValueError(
            "resume_from_checkpoint, continuation_checkpoint, "
            "stage_transition_checkpoint, and "
            "discriminator_expansion_checkpoint are mutually exclusive"
        )
    if (
        getattr(args, "stage_transition_checkpoint", None) is not None
        and not args.gan_enabled
    ):
        raise ValueError("stage_transition_checkpoint requires GAN-enabled training")
    if getattr(args, "discriminator_expansion_checkpoint", None) is not None:
        if not args.gan_enabled:
            raise ValueError(
                "discriminator_expansion_checkpoint requires GAN-enabled training"
            )
        if args.image_gan_weight <= 0 or args.video_gan_weight <= 0:
            raise ValueError(
                "discriminator expansion requires positive image and video GAN weights"
            )
    if args.sequence_length != 4:
        raise ValueError("StereoVAE training requires sequence_length=4")
    if not 0 <= args.single_frame_source_index < args.sequence_length:
        raise ValueError("--single_frame_source_index must be in [0, 3]")
    if args.resolution != 256:
        raise ValueError("the frozen pilot recipe requires resolution=256")
    if args.four_mode_mixed_training:
        if args.single_frame_source_index != 0:
            raise ValueError(
                "four-mode training currently requires "
                "--single_frame_source_index=0"
            )
        _validate_four_mode_batch_contract(args)
        if args.mode_updates_per_epoch < 1:
            raise ValueError("mode_updates_per_epoch must be positive")
        if args.mode_schedule_start_update < 0:
            raise ValueError("mode_schedule_start_update must be non-negative")
        if (
            getattr(args, "resume_from_checkpoint", None) is None
            and getattr(args, "continuation_checkpoint", None) is None
            and getattr(args, "stage_transition_checkpoint", None) is None
            and getattr(args, "discriminator_expansion_checkpoint", None) is None
            and args.mode_schedule_start_update != 0
        ):
            raise ValueError("mode_schedule_start_update requires a checkpoint")
        remaining_updates = args.max_steps - args.mode_schedule_start_update
        if remaining_updates < 1:
            raise ValueError("resume checkpoint has already reached max_steps")
        if args.mode_updates_per_epoch < remaining_updates:
            raise ValueError(
                "mode_updates_per_epoch must cover all remaining updates so the "
                "stateless resume schedule stays in one data epoch"
            )
        required_sources = {
            "hy_manifest": args.hy_manifest,
            "hy_root_aliases": args.hy_root_aliases,
            "libero_manifest": args.libero_manifest,
            "libero_root_aliases": args.libero_root_aliases,
            "umi_manifest": args.umi_manifest,
            "umi_dataset_root": args.umi_dataset_root,
            "umi_rectification_audit_sha256": (
                args.umi_rectification_audit_sha256
            ),
            "da3_repo": args.da3_repo,
            "da3_source_sha": args.da3_source_sha,
            "da3_checkpoint": args.da3_checkpoint,
            "da3_checkpoint_sha256": args.da3_checkpoint_sha256,
        }
        missing = [name for name, value in required_sources.items() if not value]
        if missing:
            raise ValueError("three-source training requires " + ", ".join(missing))
        if not args.online_gt_enabled:
            raise ValueError("four-mode online-teacher smoke requires online GT")
        if args.da3_process_res != 504:
            raise ValueError("DA3-BASE smoke process resolution is frozen to 504")
        if args.da3_confidence_mask_mode != "finite_positive_non_padding":
            raise ValueError("formal DA3 confidence threshold is not frozen")
        for name, value, length in (
            ("da3_source_sha", args.da3_source_sha, 40),
            ("da3_checkpoint_sha256", args.da3_checkpoint_sha256, 64),
            (
                "umi_rectification_audit_sha256",
                args.umi_rectification_audit_sha256,
                64,
            ),
        ):
            if len(value) != length:
                raise ValueError(f"{name} must be a full hexadecimal digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error
    geometry_values = {
        "stereo_disparity_min_px": args.stereo_disparity_min_px,
        "stereo_disparity_max_px": args.stereo_disparity_max_px,
        "stereo_lr_error_abs_threshold_px": (
            args.stereo_lr_error_abs_threshold_px
        ),
        "stereo_lr_error_relative_threshold": (
            args.stereo_lr_error_relative_threshold
        ),
    }
    missing_geometry = [
        name for name, value in geometry_values.items() if value is None
    ]
    if missing_geometry:
        raise ValueError(
            "stereo supervision requires " + ", ".join(missing_geometry)
        )
    if not 0 <= args.stereo_disparity_min_px < args.stereo_disparity_max_px:
        raise ValueError("invalid disparity supervision range")
    if args.stereo_lr_error_abs_threshold_px < 0:
        raise ValueError("absolute LR threshold must be non-negative")
    if args.stereo_lr_error_relative_threshold < 0:
        raise ValueError("relative LR threshold must be non-negative")
    teacher_sha256 = (
        args.las2_h_checkpoint_sha256
        if args.foundation_stereo_backend == "las2_h"
        else args.foundation_stereo_checkpoint_sha256
    )
    required = {"teacher_checkpoint_sha256": teacher_sha256}
    if not args.four_mode_mixed_training:
        required.update(
            {
                "lerobot_episode_manifest": args.lerobot_episode_manifest,
                "lerobot_dataset_root": args.lerobot_dataset_root,
                "lerobot_rectification_audit_sha256": (
                    args.lerobot_rectification_audit_sha256
                ),
            }
        )
    if args.foundation_stereo_backend == "las2_h":
        required.update(
            {
                "las2_h_repo": args.las2_h_repo,
                "las2_h_source_sha": args.las2_h_source_sha,
                "las2_h_checkpoint": args.las2_h_checkpoint,
                "las2_h_checkpoint_sha256": args.las2_h_checkpoint_sha256,
            }
        )
    elif args.foundation_stereo_backend == "pytorch":
        required.update(
            {
                "foundation_stereo_repo": args.foundation_stereo_repo,
                "foundation_stereo_checkpoint": (
                    args.foundation_stereo_checkpoint
                ),
            }
        )
    else:
        required.update(
            {
                "foundation_stereo_engine": args.foundation_stereo_engine,
                "foundation_stereo_engine_sha256": (
                    args.foundation_stereo_engine_sha256
                ),
                "foundation_stereo_engine_manifest": (
                    args.foundation_stereo_engine_manifest
                ),
                "foundation_stereo_engine_manifest_sha256": (
                    args.foundation_stereo_engine_manifest_sha256
                ),
            }
        )
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "LeRobot online training requires " + ", ".join(missing)
        )
    if not args.online_gt_enabled:
        raise ValueError("LeRobot online training requires --online_gt_enabled=1")
    if not args.four_mode_mixed_training and len(
        args.lerobot_rectification_audit_sha256
    ) != 64:
        raise ValueError("a full rectification audit SHA256 is required")
    if len(teacher_sha256) != 64:
        raise ValueError("a full online teacher checkpoint SHA256 is required")
    if args.online_gt_cache_enabled and not args.online_gt_cache_root:
        raise ValueError("online GT cache requires --online_gt_cache_root")
    if args.foundation_stereo_pair_microbatch < 1:
        raise ValueError("FoundationStereo pair microbatch must be positive")
    engine_values = (
        args.foundation_stereo_engine,
        args.foundation_stereo_engine_sha256,
        args.foundation_stereo_engine_manifest,
        args.foundation_stereo_engine_manifest_sha256,
    )
    if args.foundation_stereo_backend == "las2_h":
        if len(args.las2_h_source_sha) != 40:
            raise ValueError("LAS2-H requires a full source Git SHA")
        try:
            int(args.las2_h_source_sha, 16)
        except ValueError as error:
            raise ValueError("LAS2-H source SHA must be hexadecimal") from error
        if args.las2_h_valid_iters < 1:
            raise ValueError("LAS2-H valid_iters must be positive")
        if args.las2_h_max_disp != 192:
            raise ValueError("LAS2-H max_disp is frozen to 192")
        if any(value is not None for value in engine_values):
            raise ValueError("LAS2-H forbids TensorRT engine arguments")
    elif args.foundation_stereo_backend == "pytorch":
        if any(value is not None for value in engine_values):
            raise ValueError(
                "PyTorch FoundationStereo forbids TensorRT engine arguments"
            )
    else:
        if args.foundation_stereo_valid_iters != 32:
            raise ValueError("TensorRT FoundationStereo is frozen to 32 iterations")
        if args.foundation_stereo_pair_microbatch > 48:
            raise ValueError(
                "TensorRT pair microbatch exceeds the frozen max batch 48"
            )
        validate_tensorrt_engine_assets(
            args.foundation_stereo_engine,
            args.foundation_stereo_engine_sha256,
            args.foundation_stereo_engine_manifest,
            args.foundation_stereo_engine_manifest_sha256,
            args.foundation_stereo_checkpoint_sha256,
        )
    if args.online_val_check_interval_steps < 1:
        raise ValueError("online validation interval must be positive")
    if args.lerobot_val_sample_limit != 512:
        raise ValueError("online validation sample count is frozen to 512")
    if args.lerobot_video_cache_capacity < 1:
        raise ValueError("LeRobot video cache capacity must be positive")
    if args.prefetch_factor < 1:
        raise ValueError("DataLoader prefetch factor must be positive")
    if args.lerobot_maximum_timestamp_error_s <= 0:
        raise ValueError("LeRobot timestamp tolerance must be positive")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.devices < 1:
        raise ValueError("--devices must be positive")
    if args.num_nodes < 1:
        raise ValueError("--num_nodes must be positive")
    validate_distributed_runtime_args(args)
    if args.max_steps < 1:
        raise ValueError("--max_steps must be positive")
    if args.checkpoint_every_n_steps < 1:
        raise ValueError("--checkpoint_every_n_steps must be positive")
    if args.step_timing_warmup < 0:
        raise ValueError("--step_timing_warmup must be non-negative")
    if (
        args.step_timing_output is not None
        and args.step_timing_warmup >= args.max_steps
    ):
        raise ValueError("step timing warmup must leave at least one measured update")
    profile_schedule_steps = (
        args.torch_profile_wait
        + args.torch_profile_warmup
        + args.torch_profile_active
    )
    if args.torch_profile_output_dir is not None:
        if min(
            args.torch_profile_wait,
            args.torch_profile_warmup,
            args.torch_profile_active,
        ) < 0 or args.torch_profile_active < 1:
            raise ValueError("invalid torch profiler schedule")
        if profile_schedule_steps >= args.max_steps:
            raise ValueError(
                "torch profiler schedule must leave a post-profile update"
            )
    if args.train_epoch_repeats != 1:
        raise ValueError("LeRobot online training requires train_epoch_repeats=1")


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
    if args.four_mode_mixed_training:
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


def build_callbacks(args):
    callbacks = []
    if getattr(args, "discriminator_expansion_checkpoint", None) is not None:
        callbacks.append(DiscriminatorExpansionOptimizerCallback())
    if args.step_timing_output is not None:
        callbacks.append(
            StepTimingCallback(args.step_timing_output, args.step_timing_warmup)
        )
    if args.online_gt_enabled:
        callbacks.append(OnlineFoundationGTCallback(args))
        if args.four_mode_mixed_training:
            callbacks.append(OnlineDepthAnything3GTCallback(args))
    callbacks.extend([
        ModelCheckpoint(
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_top_k=-1,
            save_last=True,
            filename="{epoch}-{step}",
        )
    ])
    if not args.disable_media_logging:
        callbacks.extend(
            [
                ImageLogger(
                    batch_frequency=args.image_log_every_n_steps,
                    max_images=4,
                    clamp=True,
                ),
                VideoLogger(
                    batch_frequency=args.video_log_every_n_steps,
                    max_videos=4,
                    clamp=True,
                ),
            ]
        )
    if not args.disable_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        ModelCheckpoint(
            monitor=(
                "val/mixed/total_loss"
                if args.four_mode_mixed_training
                else "val/four/total_loss"
            ),
            every_n_epochs=1,
            save_top_k=3,
            mode="min",
            filename="best-{epoch}-{step}",
        )
    )
    return callbacks


def _load_stage_transition_checkpoint(model, checkpoint, checkpoint_path):
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("stage transition checkpoint has no model state_dict")
    discriminator_prefixes = ("image_discriminator.", "video_discriminator.")
    source_discriminator_keys = {
        key
        for key in state_dict
        if key.startswith(discriminator_prefixes)
    }
    if source_discriminator_keys:
        raise ValueError(
            "stage transition source already contains discriminator weights; "
            "use strict resume"
        )
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith(discriminator_prefixes)
    }
    if not expected_missing:
        raise ValueError("stage transition target has no discriminator parameters")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise ValueError(
            "stage transition model mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    counters = checkpoint.get("stereo_update_counters")
    if not isinstance(counters, dict) or counters.get("discriminator_updates") != 0:
        raise ValueError(
            "stage transition requires a GAN-free checkpoint with zero "
            "discriminator updates"
        )
    model.on_load_checkpoint(checkpoint)
    model.stage_transition_source = str(Path(checkpoint_path).resolve())


def _load_continuation_checkpoint(model, checkpoint, checkpoint_path):
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("continuation checkpoint has no model state_dict")
    model.load_state_dict(state_dict, strict=True)
    source = checkpoint.get("stereo_update_counters")
    if not isinstance(source, dict):
        raise ValueError("continuation checkpoint has no stereo update counters")
    if source.get("logical_update_contract_version") != 1:
        raise ValueError("continuation source must use logical-update contract v1")
    if source.get("discriminator_updates") != 0 or model.gan_enabled:
        raise ValueError("continuation requires GAN-free source and target")
    mode_weights = parse_weight_spec(model.args.mode_update_weights, MODE_IDS)
    if source.get("mode_contract") != list(MODE_IDS):
        raise ValueError("continuation source four-mode contract mismatch")
    if source.get("mode_update_weights") != mode_weights:
        raise ValueError("continuation source mode weights mismatch")
    if source.get("mono_dataset_weights") != model.args.mono_dataset_weights:
        raise ValueError("continuation source mono dataset weights mismatch")
    if source.get("mode_schedule_seed") != model.args.mode_schedule_seed:
        raise ValueError("continuation source mode schedule seed mismatch")
    generator_updates = source.get("generator_updates")
    mode_updates = source.get("mode_updates")
    mode_samples = source.get("mode_samples")
    if type(generator_updates) is not int or generator_updates < 0:
        raise ValueError("continuation source generator counter is invalid")
    if not isinstance(mode_updates, dict) or set(mode_updates) != set(MODE_IDS):
        raise ValueError("continuation source mode counters mismatch")
    if not isinstance(mode_samples, dict) or set(mode_samples) != set(MODE_IDS):
        raise ValueError("continuation source sample counters mismatch")
    expected_mode_updates = mode_occurrences_before(
        model.args.mode_schedule_seed, generator_updates, mode_weights
    )
    if mode_updates != expected_mode_updates:
        raise ValueError("continuation source counters disagree with schedule")
    world_size = int(model.args.devices * model.args.num_nodes)
    transition = {
        "source_generator_updates": generator_updates,
        "source_batch_updates": int(source["batch_updates"]),
        "source_mode_updates": dict(mode_updates),
        "source_mode_samples": dict(mode_samples),
        "source_contract": {
            key: source.get(key)
            for key in (
                "node_manifest_contracts",
                "per_device_batch_size",
                "grad_accumulates",
                "mode_batch_sizes",
                "mode_grad_accumulates",
                "mode_effective_global_batch_sizes",
                "world_size_contract",
            )
        },
    }
    adapted = dict(source)
    adapted.update(
        {
            "node_manifest_contracts": model.args.node_manifest_contracts,
            "per_device_batch_size": int(model.args.batch_size),
            "grad_accumulates": int(model.grad_accumulates),
            "mode_batch_sizes": dict(model.mode_batch_sizes),
            "mode_grad_accumulates": dict(model.mode_grad_accumulates),
            "mode_effective_global_batch_sizes": {
                mode_id: model.mode_batch_sizes[mode_id]
                * model.mode_grad_accumulates[mode_id]
                * world_size
                for mode_id in MODE_IDS
            },
            "logical_update_contract_version": 2,
            "world_size_contract": world_size,
            "counter_transition": transition,
        }
    )
    model.on_load_checkpoint({"stereo_update_counters": adapted})
    model.continuation_source = str(Path(checkpoint_path).resolve())


def _load_discriminator_expansion_checkpoint(model, checkpoint, checkpoint_path):
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("discriminator expansion checkpoint has no model state_dict")
    source_image_keys = {
        key for key in state_dict if key.startswith("image_discriminator.")
    }
    source_video_keys = {
        key for key in state_dict if key.startswith("video_discriminator.")
    }
    if not source_image_keys:
        raise ValueError(
            "discriminator expansion source has no image discriminator weights"
        )
    if source_video_keys:
        raise ValueError(
            "discriminator expansion source already has video discriminator weights"
        )
    target_state = model.state_dict()
    target_image_keys = {
        key for key in target_state if key.startswith("image_discriminator.")
    }
    target_video_keys = {
        key for key in target_state if key.startswith("video_discriminator.")
    }
    if source_image_keys != target_image_keys or not target_video_keys:
        raise ValueError(
            "discriminator expansion target topology does not preserve image and add video"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != target_video_keys or unexpected:
        raise ValueError(
            "discriminator expansion model mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    counters = checkpoint.get("stereo_update_counters")
    if not isinstance(counters, dict) or counters.get("discriminator_updates", 0) <= 0:
        raise ValueError(
            "discriminator expansion requires a GAN checkpoint with positive "
            "discriminator updates"
        )
    optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 2:
        raise ValueError(
            "discriminator expansion requires generator and discriminator optimizer states"
        )
    model.on_load_checkpoint(checkpoint)
    model._discriminator_expansion_optimizer_states = optimizer_states
    model.discriminator_expansion_source = str(Path(checkpoint_path).resolve())


def main():
    parser = build_parser()
    args = parser.parse_args()
    checkpoint_args = (
        args.resume_from_checkpoint,
        args.continuation_checkpoint,
        args.stage_transition_checkpoint,
        args.discriminator_expansion_checkpoint,
    )
    if sum(value is not None for value in checkpoint_args) > 1:
        raise ValueError(
            "resume_from_checkpoint, continuation_checkpoint, "
            "stage_transition_checkpoint, and "
            "discriminator_expansion_checkpoint are mutually exclusive"
        )
    checkpoint_path = next(
        (value for value in checkpoint_args if value is not None), None
    )
    checkpoint = None
    counters = None
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        counters = checkpoint.get("stereo_update_counters")
        if not isinstance(counters, dict):
            raise ValueError("resume checkpoint has no stereo update counters")
        args.mode_schedule_start_update = int(counters["generator_updates"])
        if args.continuation_checkpoint is not None:
            digest = hashlib.sha256()
            with args.continuation_checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            args.continuation_checkpoint_sha256 = digest.hexdigest()
            args.continuation_source_generator_updates = int(
                counters["generator_updates"]
            )
            args.continuation_source_contract = {
                key: counters.get(key)
                for key in (
                    "node_manifest_contracts",
                    "per_device_batch_size",
                    "grad_accumulates",
                    "mode_batch_sizes",
                    "mode_grad_accumulates",
                    "mode_effective_global_batch_sizes",
                    "world_size_contract",
                )
            }
    _bind_node_manifest_contracts(args)
    validate_runtime_args(args)
    if checkpoint_path is not None and args.four_mode_mixed_training:
        if counters.get("mode_schedule_seed") != args.mode_schedule_seed:
            raise ValueError("resume checkpoint mode schedule seed mismatch")
        if counters.get("mode_updates") != mode_occurrences_before(
            args.mode_schedule_seed,
            args.mode_schedule_start_update,
            parse_weight_spec(args.mode_update_weights, MODE_IDS),
        ):
            raise ValueError("resume checkpoint counters disagree with next mode")
    write_online_gt_run_metadata(args)
    pl.seed_everything(args.seed)

    data = StereoDataModule(args)
    model = StereoVAE(args)
    if args.continuation_checkpoint is not None:
        _load_continuation_checkpoint(
            model,
            checkpoint,
            args.continuation_checkpoint,
        )
    elif args.stage_transition_checkpoint is not None:
        _load_stage_transition_checkpoint(
            model,
            checkpoint,
            args.stage_transition_checkpoint,
        )
    elif args.discriminator_expansion_checkpoint is not None:
        _load_discriminator_expansion_checkpoint(
            model,
            checkpoint,
            args.discriminator_expansion_checkpoint,
        )
    checkpoint = None
    callbacks = build_callbacks(args)

    profiler = None
    if args.torch_profile_output_dir is not None and int(
        os.environ.get("LOCAL_RANK", "0")
    ) == 0:
        set_profiling_enabled(True)
        trace_writer = TrainingTraceWriter(
            args.torch_profile_output_dir,
            args.torch_profile_active,
        )
        profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=args.torch_profile_wait,
                warmup=args.torch_profile_warmup,
                active=args.torch_profile_active,
                repeat=1,
            ),
            on_trace_ready=trace_writer,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        callbacks.append(TrainingProfilerStepCallback(profiler))

    logger = False
    if not args.disable_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=os.path.basename(os.path.abspath(args.default_root_dir)),
            save_dir=args.default_root_dir,
            config=vars(args),
        )

    precision = "32-true"
    if args.bf16:
        precision = "bf16-mixed"
    elif args.fp16:
        precision = "16-mixed"

    strategy = "auto"
    if args.devices * args.num_nodes > 1:
        strategy = DDPStrategy(
            static_graph=False,
            find_unused_parameters=True,
        )

    val_check_interval = _resolve_val_check_interval(args)
    check_val_every_n_epoch = None

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        max_steps=(
            -1
            if args.gan_enabled
            or getattr(args, "continuation_checkpoint", None) is not None
            else args.max_steps
        ),
        max_epochs=-1,
        default_root_dir=args.default_root_dir,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        use_distributed_sampler=False,
    )
    with profiler if profiler is not None else nullcontext():
        trainer.fit(
            model,
            datamodule=data,
            ckpt_path=args.resume_from_checkpoint,
        )


if __name__ == "__main__":
    main()
