"""H200-2 native-test latent ablation; experiment only.
Metric sums reuse eval_stereo_vae.py at cfa07e2^, with RGB content cropping.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure
from evaluation.stage_a import runtime
from evaluation.stage_a.runtime import _relative_target_from_batch
from stereo_tokenizer.modules.relative_depth import relative_prediction_from_raw
from stereo_tokenizer.pretrain_data import HyLanceMonoDataset, LiberoMonoDataset
from stereo_tokenizer.lerobot_data import LeRobotStereoDataset
from stereo_tokenizer.online_gt import sha256_file

def empty_accumulator(device, view_count):
    return {
        "sample_count": torch.zeros((), dtype=torch.long, device=device),
        "rgb_abs_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_sq_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_count": torch.zeros((), dtype=torch.long, device=device),
        "rgb_frame_psnr_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_ssim_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_lpips_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_frame_count": torch.zeros((), dtype=torch.long, device=device),
        "rgb_lpips_frame_count": torch.zeros(
            (), dtype=torch.long, device=device
        ),
        "temporal_delta_abs_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "temporal_delta_count": torch.zeros((), dtype=torch.long, device=device),
        "temporal_delta_lpips_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "temporal_pair_count": torch.zeros((), dtype=torch.long, device=device),
        "temporal_lpips_pair_count": torch.zeros(
            (), dtype=torch.long, device=device
        ),
        "relative_log_abs_sum": torch.zeros(
            view_count, dtype=torch.float64, device=device
        ),
        "relative_log_sum": torch.zeros(
            view_count, dtype=torch.float64, device=device
        ),
        "valid_count": torch.zeros(view_count, dtype=torch.long, device=device),
        "relative_log_sq_sum": torch.zeros(
            view_count, dtype=torch.float64, device=device
        ),
    }


def _flatten_rgb_frames(video):
    return video.permute(0, 1, 3, 2, 4, 5).reshape(-1, 3, *video.shape[-2:])


def _lpips_sum(perceptual_model, prediction, target, microbatch):
    if perceptual_model is None:
        raise RuntimeError("LPIPS evaluation requires the checkpoint perceptual model")
    total = prediction.new_zeros((), dtype=torch.float64)
    for start in range(0, len(prediction), microbatch):
        stop = min(start + microbatch, len(prediction))
        total += perceptual_model(
            prediction[start:stop].float(), target[start:stop].float()
        ).double().sum()
    return total


def _ssim_sum(prediction, target, microbatch):
    total = prediction.new_zeros((), dtype=torch.float64)
    for start in range(0, len(prediction), microbatch):
        stop = min(start + microbatch, len(prediction))
        total += structural_similarity_index_measure(
            prediction[start:stop].add(0.5).clamp(0, 1),
            target[start:stop].add(0.5).clamp(0, 1),
            data_range=1.0,
            reduction="sum",
        ).double()
    return total


def update_metrics(
    accumulator,
    batch,
    output,
    relative_depth_epsilon,
    perceptual_model=None,
    metric_frame_microbatch=24,
):
    rgb_target = batch["video"][:, :, 0].float()
    rgb_prediction = output.rgb.float()
    mask = batch.get("non_padding_mask")
    if mask is not None:
        spatial = mask.reshape(-1, *mask.shape[-2:]).bool()
        if not torch.equal(spatial, spatial[:1].expand_as(spatial)):
            raise ValueError("RGB content masks differ within batch")
        positions = spatial[0].nonzero()
        lo = positions.min(0).values.tolist()
        hi = (positions.max(0).values + 1).tolist()
        if int(spatial[0].sum()) != (hi[0]-lo[0])*(hi[1]-lo[1]):
            raise ValueError("RGB content mask is not rectangular")
        rgb_target = rgb_target[..., lo[0]:hi[0], lo[1]:hi[1]]
        rgb_prediction = rgb_prediction[..., lo[0]:hi[0], lo[1]:hi[1]]
    expected_views = int(accumulator["valid_count"].numel())
    if int(rgb_target.shape[1]) != expected_views:
        raise ValueError(
            f"evaluation accumulator expects {expected_views} views, "
            f"batch contains {int(rgb_target.shape[1])}"
        )
    rgb_error = (rgb_prediction - rgb_target).abs()
    accumulator["sample_count"] += rgb_target.shape[0]
    accumulator["rgb_abs_sum"] += rgb_error.double().sum()
    accumulator["rgb_sq_sum"] += rgb_error.square().double().sum()
    accumulator["rgb_count"] += rgb_error.numel()

    prediction_frames = _flatten_rgb_frames(rgb_prediction)
    target_frames = _flatten_rgb_frames(rgb_target)
    frame_mse = (prediction_frames - target_frames).square().mean((1, 2, 3))
    accumulator["rgb_frame_psnr_sum"] += (
        -10.0 * torch.log10(frame_mse.clamp_min(1e-12))
    ).double().sum()
    accumulator["rgb_ssim_sum"] += _ssim_sum(
        prediction_frames,
        target_frames,
        metric_frame_microbatch,
    )
    if perceptual_model is not None:
        accumulator["rgb_lpips_sum"] += _lpips_sum(
            perceptual_model,
            prediction_frames * 2.0,
            target_frames * 2.0,
            metric_frame_microbatch,
        )
        accumulator["rgb_lpips_frame_count"] += len(prediction_frames)
    accumulator["rgb_frame_count"] += len(prediction_frames)

    if rgb_target.shape[3] > 1:
        target_delta = rgb_target[:, :, :, 1:] - rgb_target[:, :, :, :-1]
        prediction_delta = (
            rgb_prediction[:, :, :, 1:] - rgb_prediction[:, :, :, :-1]
        )
        delta_error = (prediction_delta - target_delta).abs()
        accumulator["temporal_delta_abs_sum"] += delta_error.double().sum()
        accumulator["temporal_delta_count"] += delta_error.numel()
        prediction_delta_frames = _flatten_rgb_frames(prediction_delta)
        target_delta_frames = _flatten_rgb_frames(target_delta)
        if perceptual_model is not None:
            accumulator["temporal_delta_lpips_sum"] += _lpips_sum(
                perceptual_model,
                prediction_delta_frames,
                target_delta_frames,
                metric_frame_microbatch,
            )
            accumulator["temporal_lpips_pair_count"] += len(
                prediction_delta_frames
            )
        accumulator["temporal_pair_count"] += len(prediction_delta_frames)

    valid = batch["valid_mask"]
    target = _relative_target_from_batch(batch, relative_depth_epsilon)
    prediction, _ = relative_prediction_from_raw(
        output.raw_relative_log_depth, valid
    )
    relative_error = prediction - target.relative_log_depth
    reduction_dims = (0, 2, 3, 4, 5)
    accumulator["relative_log_abs_sum"] += (
        relative_error.abs().double().masked_fill(~valid, 0).sum(dim=reduction_dims)
    )
    accumulator["relative_log_sum"] += (
        relative_error.double().masked_fill(~valid, 0).sum(dim=reduction_dims)
    )
    accumulator["valid_count"] += valid.sum(dim=reduction_dims)
    accumulator["relative_log_sq_sum"] += (
        relative_error.square()
        .double()
        .masked_fill(~valid, 0)
        .sum(dim=reduction_dims)
    )


def finalize_metrics(accumulator, view_names):
    if len(view_names) != int(accumulator["valid_count"].numel()):
        raise ValueError("view names disagree with evaluation accumulator")
    if accumulator["sample_count"].item() == 0:
        raise ValueError("evaluation loader produced no samples")
    if torch.any(accumulator["valid_count"] == 0):
        raise ValueError("at least one view has no valid relative-depth pixels")
    valid_count = accumulator["valid_count"].double()
    result = {
        "sample_count": int(accumulator["sample_count"].item()),
        "rgb_l1": float(
            (accumulator["rgb_abs_sum"] / accumulator["rgb_count"]).item()
        ),
        "rgb_psnr_db_global": float(
            (
                -10.0
                * torch.log10(
                    accumulator["rgb_sq_sum"]
                    / accumulator["rgb_count"]
                )
            ).item()
        ),
        "rgb_psnr_db_frame_mean": float(
            (
                accumulator["rgb_frame_psnr_sum"]
                / accumulator["rgb_frame_count"]
            ).item()
        ),
        "rgb_ssim_frame_mean": float(
            (
                accumulator["rgb_ssim_sum"]
                / accumulator["rgb_frame_count"]
            ).item()
        ),
        "valid_pixels": int(accumulator["valid_count"].sum().item()),
        "views": {},
    }
    if accumulator["rgb_lpips_frame_count"].item():
        result["rgb_lpips_frame_mean"] = float(
            (
                accumulator["rgb_lpips_sum"]
                / accumulator["rgb_lpips_frame_count"]
            ).item()
        )
    if accumulator["temporal_pair_count"].item():
        result["temporal_delta_l1"] = float(
            (
                accumulator["temporal_delta_abs_sum"]
                / accumulator["temporal_delta_count"]
            ).item()
        )
    if accumulator["temporal_lpips_pair_count"].item():
        result["temporal_delta_lpips_frame_mean"] = float(
            (
                accumulator["temporal_delta_lpips_sum"]
                / accumulator["temporal_lpips_pair_count"]
            ).item()
        )
    for view_index, view_name in enumerate(view_names):
        result["views"][view_name] = {
            "valid_pixels": int(accumulator["valid_count"][view_index].item()),
            "relative_log_l1": float(
                (
                    accumulator["relative_log_abs_sum"][view_index]
                    / valid_count[view_index]
                ).item()
            ),
            "relative_log_rmse": float(
                torch.sqrt(
                    accumulator["relative_log_sq_sum"][view_index]
                    / valid_count[view_index]
                ).item()
            ),
            "relative_log_silog": float(
                torch.sqrt(
                    torch.clamp_min(
                        accumulator["relative_log_sq_sum"][view_index]
                        / valid_count[view_index]
                        - (
                            accumulator["relative_log_sum"][view_index]
                            / valid_count[view_index]
                        ).square(),
                        0.0,
                    )
                ).item()
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hy", "libero", "umi"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    opt = parser.parse_args()
    if opt.output.exists():
        raise FileExistsError(opt.output)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = {z: json.loads((opt.run_root / f"z{z}/resolved_config.json").read_text())
               for z in (24, 48, 96)}
    for z in (48,96):
        assert {k:v for k,v in configs[z].items() if k not in ("latent_channels","default_root_dir")} == {
            k:v for k,v in configs[24].items() if k not in ("latent_channels","default_root_dir")}
    cfg = configs[24]
    args = argparse.Namespace(**cfg)
    for k, v in dict(foundation_stereo_engine=None, foundation_stereo_engine_sha256=None,
                     foundation_stereo_engine_manifest=None, foundation_stereo_engine_manifest_sha256=None).items():
        if not hasattr(args,k): setattr(args,k,v)
    args.single_frame_source_indices = [0,1,2,3]
    eye = "stereo" if opt.dataset == "umi" else "mono"
    if opt.dataset == "hy":
        dataset = HyLanceMonoDataset(cfg["hy_manifest"],json.loads(cfg["hy_root_aliases"]),split="test",single_frame_source_index=0)
        views = ("cam_high","cam_left_wrist","cam_right_wrist")
    elif opt.dataset == "libero":
        dataset = LiberoMonoDataset(cfg["libero_manifest"],json.loads(cfg["libero_root_aliases"]),split="test",single_frame_source_index=0)
        views = ("agentview","wrist")
    else:
        dataset = LeRobotStereoDataset(cfg["umi_manifest"],cfg["umi_dataset_root"],split="test",
            expected_rectification_audit_sha256=cfg["umi_rectification_audit_sha256"],single_frame_source_index=0)
        views = ("head","lefthand","righthand")
    manifest = Path(cfg[opt.dataset+"_manifest"])
    expected_hash = json.loads(cfg["node_manifest_contracts"])["0"][opt.dataset]
    assert sha256_file(manifest) == expected_hash
    indices = list(range(rank, len(dataset), world))
    if opt.max_batches:
        indices = indices[:opt.max_batches*opt.batch_size]
    loader = DataLoader(Subset(dataset,indices),batch_size=opt.batch_size,num_workers=4,
                        pin_memory=True,shuffle=False,drop_last=False)
    models = {}
    provenance = {}
    for z in (24,48,96):
        ckpt = next((opt.run_root/f"z{z}").glob("stereo-vae/*/checkpoints/epoch=0-step=40000.ckpt"))
        args.stereo_vae_ckpt = ckpt
        args.latent_channels = z
        models[z] = runtime.load_model(args,device).requires_grad_(False)
        provenance[z] = {"checkpoint":str(ckpt),"sha256":sha256_file(ckpt)}
    runtime.preflight_teacher_assets(args,(eye,))
    teacher = runtime.build_online_teacher(args,eye,device)
    specs = runtime.evaluation_specs(args,eye)
    accum = {z:{mode:empty_accumulator(device,len(views)) for mode,_,_ in specs} for z in models}
    ids_hash = hashlib.sha256()
    targets_hash = hashlib.sha256()
    started = time.monotonic()
    print(json.dumps({"event":"ready","rank":rank,"dataset":opt.dataset,"full_samples":len(dataset),
                      "rank_samples":len(indices),"batches":len(loader),"batch_size":opt.batch_size}),flush=True)
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            ids_hash.update(json.dumps(batch["sample_id"],sort_keys=True).encode())
            batch = {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            runtime.attach_online_targets(args,eye,teacher,batch)
            for key in ("valid_mask","disparity","da3_relative_depth"):
                if key in batch: targets_hash.update(batch[key].contiguous().cpu().numpy().tobytes())
            for mode,temporal,source in specs:
                b = runtime.batch_for_temporal_mode(batch,temporal,source)
                for z,model in models.items():
                    output = model(b["video"],eye_mode=eye,temporal_mode=temporal,sample_posterior=False)
                    update_metrics(accum[z][mode],b,output,args.relative_depth_epsilon,
                                   model.perceptual_model,metric_frame_microbatch=12)
                    del output
            if i % 10 == 0 or i+1 == len(loader):
                print(json.dumps({"event":"progress","rank":rank,"batch":i+1,"batches":len(loader),
                    "elapsed_s":time.monotonic()-started,
                    "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
    hashes = [None]*world
    dist.all_gather_object(hashes,{"rank":rank,"sample_count":len(indices),
                                 "sample_ids_sha256":ids_hash.hexdigest(),"targets_sha256":targets_hash.hexdigest()})
    for z in accum:
        for mode in accum[z]:
            for value in accum[z][mode].values(): dist.all_reduce(value)
    if rank == 0:
        metrics = {z:{mode:finalize_metrics(a,views) for mode,a in modes.items()} for z,modes in accum.items()}
        expected = sum(h["sample_count"] for h in hashes)
        if opt.max_batches is None: assert expected == len(dataset)
        for modes in metrics.values():
            for values in modes.values():
                assert values["sample_count"] == expected
                values["macro_relative_log_l1"] = sum(v["relative_log_l1"] for v in values["views"].values())/len(views)
        result = {"dataset":opt.dataset,"split":"test","full_split":opt.max_batches is None,
          "sample_count":expected,"manifest":str(manifest),"manifest_sha256":expected_hash,
          "precision":"FP32; TF32 disabled","posterior":"mean","batch_size":opt.batch_size,
          "world_size":world,"sample_target_hashes":hashes,"checkpoints":provenance,
          "rgb_region":"rectangular non-padding content","geometry":"teacher-relative centered log depth",
          "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
          "script_sha256":sha256_file(Path(__file__)),"metrics":metrics,
          "elapsed_s":time.monotonic()-started}
        opt.output.parent.mkdir(parents=True,exist_ok=True)
        opt.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"event":"complete","output":str(opt.output),"samples":expected}),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
