"""Paired RGB/temporal evaluation using the existing Stage A metric suite."""

import hashlib
import json
import time

import torch
from torch.utils.data import DataLoader

from evaluation.stage_a import runtime
from evaluation.stage_a.common import _FrozenRAFT, _checkpoint_provenance, _dataset_provenance, _environment_provenance, _jsonable, _source_provenance
from evaluation.stage_a.data import CanonicalStageADataset
from evaluation.stage_a.metrics import StageA1MetricSuite
from evaluation.stage_a.quality import _run_parser, _hydrate_checkpoint_semantics, _validate_run, _mode_batch
from .adapter import WanReconstructor, SOURCE_SHA, WEIGHT_SHA


def main():
    parser = _run_parser()
    parser.add_argument("--wan-source-root", required=True)
    parser.add_argument("--wan-checkpoint", required=True)
    args = parser.parse_args()
    if not args.rgb_only or args.num_visualizations:
        raise ValueError("paired evaluation requires --rgb-only and --num_visualizations 0")
    _hydrate_checkpoint_semantics(args)
    _validate_run(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = _source_provenance()
    if source["git_status_porcelain"]:
        raise ValueError("formal evaluation requires clean source")
    environment = _environment_provenance()
    checkpoint = _checkpoint_provenance(args.stereo_vae_ckpt, args.checkpoint_sha256)
    if checkpoint["stereo_update_counters"]["discriminator_updates"] != 0:
        raise ValueError("comparison requires GAN-off checkpoint")
    dataset = CanonicalStageADataset(args.stage_a_selection, loader_root=args.canonical_loader_root, eye_mode=args.eval_eye_mode, camera_key=args.stage_a_camera_key, hy_root_aliases=args.hy_root_aliases)
    if dataset.dataset_id != args.stage_a_dataset_id:
        raise ValueError("dataset identity mismatch")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda")
    baseline = runtime.load_model(args, device).requires_grad_(False)
    wan = WanReconstructor(args.wan_source_root, args.wan_checkpoint, device)
    flow = _FrozenRAFT(args.raft_checkpoint, args.raft_checkpoint_sha256, device=device, microbatch=args.raft_microbatch)
    suites = {name: StageA1MetricSuite(relative_depth_epsilon=args.relative_depth_epsilon) for name in ("gan_off", "wan22")}
    specs = runtime.evaluation_specs(args, args.eval_eye_mode)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_json.with_suffix(".samples.jsonl")
    started = time.monotonic()
    count = 0
    torch.cuda.reset_peak_memory_stats()
    with audit_path.open("x", encoding="utf-8") as audit, torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            for row, sample_id in enumerate(batch["sample_id"]):
                digest = hashlib.sha256()
                for key in ("video", "rgb_valid_mask"):
                    digest.update(batch[key][row].contiguous().numpy().tobytes())
                audit.write(json.dumps({"sample_id": sample_id, "input_target_mask_sha256": digest.hexdigest()}) + "\n")
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for mode_id, temporal, frame in specs:
                mode = _mode_batch(tensors, temporal, frame)
                outputs = {
                    "gan_off": baseline(mode["video"], eye_mode=args.eval_eye_mode, temporal_mode=temporal, sample_posterior=False),
                    "wan22": wan(mode["video"][:, :, 0]),
                }
                for name, output in outputs.items():
                    metric_batch = mode if name == "gan_off" else {**mode, "video": mode["video"][:, :, :1]}
                    suites[name].update(mode_id, metric_batch, output, dataset.view_names, baseline.perceptual_model, flow)
            count += len(batch["sample_id"])
            audit.flush()
            if index % 10 == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({"samples": count, "total": len(dataset), "elapsed_s": elapsed, "samples_per_s": count / elapsed, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
    if args.max_batches is None and count != len(dataset):
        raise RuntimeError("incomplete full selection")
    metrics = {name: {mode: suite.finalize(mode, dataset.view_names) for mode, _, _ in specs} for name, suite in suites.items()}
    for arm in metrics.values():
        if any(values["sample_count"] != count for values in arm.values()):
            raise RuntimeError("paired metric sample counts disagree")
    result = {
        "status": "smoke" if args.max_batches is not None else "formal", "sample_count": count,
        "dataset": _dataset_provenance(dataset), "checkpoint": checkpoint,
        "wan": {"source_sha": SOURCE_SHA, "checkpoint_sha256": WEIGHT_SHA, "latent_shapes_per_view": wan.latent_shapes, "temporal_adapter": "repeat_last_4_to_5_decode_crop_first_4", "output": "official clamp [-1,1] then /2; raw overshoot is not a cross-model quality score"},
        "precision": "fp32_tf32_off_posterior_mean", "metrics": metrics, "flow_teacher": flow.provenance(),
        "elapsed_s": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "provenance": {**source, "environment": environment, "resolved_args": _jsonable(vars(args))},
    }
    with args.output_json.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(f"COMPLETED {count} paired samples: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
