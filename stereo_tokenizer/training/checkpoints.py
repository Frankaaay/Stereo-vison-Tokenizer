"""Explicit checkpoint transition loaders and contract validation."""

from __future__ import annotations

from pathlib import Path

from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    mode_occurrences_before,
    parse_weight_spec,
)


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
