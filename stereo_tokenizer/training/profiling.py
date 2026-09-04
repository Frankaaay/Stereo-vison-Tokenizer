"""Training timing and torch-profiler callbacks."""

from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from stereo_tokenizer.mode_sampling import MODE_IDS


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
