import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "stereo_tokenizer" / "training"


def _training_source(*names: str) -> str:
    paths = (ROOT / "train_stereo_vae.py",) + tuple(
        TRAINING_ROOT / name for name in names
    )
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


class StereoEntrypointSourceTest(unittest.TestCase):
    def test_training_entry_has_no_legacy_inflation_and_supports_explicit_resume(self):
        source = _training_source("runtime.py", "checkpoints.py")
        self.assertNotIn("inflate_gen", source)
        self.assertNotIn("inflate_dis", source)
        self.assertNotIn("os.listdir", source)
        self.assertIn("StereoVAE(args)", source)
        self.assertIn("StereoDataModule(args)", source)
        self.assertIn("limit_val_batches=1.0", source)
        self.assertIn("check_val_every_n_epoch = None", source)
        self.assertIn("check_val_every_n_epoch=check_val_every_n_epoch", source)
        self.assertIn('getattr(args, "continuation_checkpoint", None)', source)
        self.assertIn("max_epochs=-1", source)
        self.assertIn("--resume_from_checkpoint", source)
        self.assertIn("--continuation_checkpoint", source)
        self.assertIn("_load_continuation_checkpoint(", source)
        self.assertIn("--stage_transition_checkpoint", source)
        self.assertIn("_load_stage_transition_checkpoint(", source)
        self.assertIn("--discriminator_expansion_checkpoint", source)
        self.assertIn("_load_discriminator_expansion_checkpoint(", source)
        self.assertIn("ckpt_path=args.resume_from_checkpoint", source)

    def test_stage_a_evaluation_is_deterministic_and_strict(self):
        entry = (ROOT / "evaluation" / "tokenizer_stage_a.py").read_text(
            encoding="utf-8"
        )
        runtime = (ROOT / "evaluation" / "stage_a" / "runtime.py").read_text(
            encoding="utf-8"
        )
        quality = (ROOT / "evaluation" / "stage_a" / "quality.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("sample_posterior=False", quality)
        self.assertIn("strict=True", runtime)
        self.assertIn("_checkpoint_model_args(checkpoint", runtime)
        self.assertIn("StereoVAE(checkpoint_args)", runtime)
        self.assertIn("--stereo_vae_ckpt", runtime)
        self.assertIn("_validate_checkpoint_semantics", runtime)
        self.assertIn("--eval_temporal_mode", runtime)
        self.assertIn("args.single_frame_source_index", quality)
        self.assertNotIn(".codebook", entry + runtime + quality)
        self.assertIn("FoundationStereoOnlineTeacher", runtime)
        self.assertIn("DepthAnything3OnlineTeacher", runtime)
        self.assertIn("relative_target_from_da3(", runtime)
        self.assertIn("save_case_visualization", runtime)
        self.assertIn("save_depth_case_visualization", runtime)
        self.assertNotIn("def evaluate_eye_mode(", runtime)
        self.assertNotIn("def update_metrics(", runtime)

    def test_recipe_requires_unfrozen_experiment_parameters(self):
        source = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        for name in (
            "PER_DEVICE_BATCH_SIZE",
            "GRAD_ACCUMULATES",
            "RGB_WEIGHT",
            "RELATIVE_DEPTH_WEIGHT",
            "RELATIVE_GRADIENT_WEIGHT",
            "KL_WEIGHT",
            "KL_WARMUP_STEPS",
            "SINGLE_FRAME_SOURCE_INDEX",
        ):
            self.assertIn(f"${{{name}:?", source)
        self.assertIn('TRAIN_LAUNCHER=(python3)', source)
        self.assertIn("python3 -m torch.distributed.run", source)
        self.assertIn(
            'PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True', source
        )
        self.assertIn(
            'DISTRIBUTED_MODE}" == "single" && "${GPU_COUNT}" -gt 1', source
        )
        self.assertIn("--standalone", source)
        self.assertIn('"${TRAIN_LAUNCHER[@]}" train_stereo_vae.py', source)
        self.assertIn('LATENT_CHANNELS="${LATENT_CHANNELS:-48}"', source)
        self.assertIn('--latent_channels "${LATENT_CHANNELS}"', source)
        self.assertIn("LATENT_CHANNELS must be 24, 48, or 96", source)
        self.assertIn(
            '--single_frame_source_index "${SINGLE_FRAME_SOURCE_INDEX}"',
            source,
        )
        self.assertIn("${LAS2_H_SOURCE_SHA:?", source)
        self.assertIn('--las2_h_source_sha "${LAS2_H_SOURCE_SHA}"', source)
        self.assertNotIn("single_frame_loss_weight", source)
        self.assertIn('GAN_ENABLED="${GAN_ENABLED:-0}"', source)
        self.assertIn("GAN_ARGS+=(--gan_enabled)", source)
        self.assertIn("STAGE_TRANSITION_CHECKPOINT", source)
        self.assertIn("DISCRIMINATOR_EXPANSION_CHECKPOINT", source)
        self.assertNotIn("--use_vae", source)
        self.assertNotIn("--fp16", source)

    def test_latent_ablation_serial_order_and_failure_contract(self):
        source = (
            ROOT / "scripts" / "stereo" / "run_latent_ablation_serial.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("for latent_channels in 96 24 48", source)
        self.assertIn('LATENT_CHANNELS="${latent_channels}"', source)
        self.assertIn('OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT}/z${latent_channels}"', source)
        self.assertIn("refusing to reuse existing ABLATION_OUTPUT_ROOT", source)
        self.assertIn('train_status=${PIPESTATUS[0]}', source)
        self.assertIn('> "${OUTPUT_ROOT}/exit_code.txt"', source)

    def test_profile_regions_are_opt_in_and_cover_requested_components(self):
        helper = (ROOT / "stereo_tokenizer" / "profiling.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_ENABLED = False", helper)
        model = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "stereo_tokenizer" / "model.py",
                ROOT / "stereo_tokenizer" / "modules" / "stereo_encoder.py",
                ROOT / "stereo_tokenizer" / "modules" / "stereo_decoder.py",
            )
        )
        data = (ROOT / "stereo_tokenizer" / "data.py").read_text(
            encoding="utf-8"
        )
        losses = (
            ROOT / "stereo_tokenizer" / "modules" / "stereo_losses.py"
        ).read_text(encoding="utf-8")
        for region in (
            "stereo/encoder/spatial_transformer",
            "stereo/encoder/stereo_fusion",
            "stereo/decoder/spatial_transformer",
            "stereo/loss/lpips_vgg",
            "stereo/update/backward",
            "stereo/update/gradient_clipping",
            "stereo/update/adam_step",
            "stereo/logging/train_metrics",
            "stereo/transfer/cpu_to_gpu",
        ):
            self.assertIn(region, model)

        self.assertIn("stereo/data/collate", data)
        self.assertNotIn("rgb_npz_read_decompress", data)
        self.assertNotIn("gt_npz_read_decompress", data)
        for region in (
            "stereo/loss/rgb",
            "stereo/loss/relative_depth",
            "stereo/loss/relative_gradient",
            "stereo/loss/kl",
        ):
            self.assertIn(region, losses)

    def test_peg_backends_are_explicit_and_training_uses_t1_conv2d(self):
        attention = (
            ROOT / "stereo_tokenizer" / "modules" / "attention.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'self._backend = "conv3d_contiguous"', attention
        )
        self.assertIn('"conv2d_t1_slice"', attention)
        launcher = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--peg_backend conv2d_t1_slice", launcher)
        self.assertIn("--pin_memory 1", launcher)
        self.assertIn("--persistent_workers 1", launcher)
        self.assertIn('--prefetch_factor "${PREFETCH_FACTOR:-2}"', launcher)
        data = (ROOT / "stereo_tokenizer" / "data.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'parser.add_argument('
            '"--lerobot_video_cache_capacity", type=int, default=12)',
            data,
        )
        self.assertIn(
            '--checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS:-500}"',
            launcher,
        )
        self.assertIn("DISABLE_MEDIA_LOGGING:-0", launcher)
        train = _training_source("profiling.py", "runtime.py")
        self.assertIn("peak_memory_bytes_by_rank", train)
        self.assertIn('"mode_id": pl_module.last_mode_id', train)
        self.assertIn("peak_memory_bytes_by_rank_and_mode", train)
        self.assertIn("DDPStrategy(", train)
        self.assertIn("static_graph=False", train)
        self.assertIn("find_unused_parameters=True", train)
        self.assertIn("--torch_profile_output_dir", train)
        self.assertIn("set_profiling_enabled(True)", train)
        self.assertIn("TrainingProfilerStepCallback", train)
        self.assertIn("ProfilerActivity.CPU", train)
        self.assertIn("ProfilerActivity.CUDA", train)
        self.assertIn('os.environ.get("LOCAL_RANK", "0")', train)
        self.assertIn("profile_memory=True", train)
        self.assertIn("record_shapes=True", train)
        self.assertNotIn('torch.__version__ >= "2.1.0"', attention)
        self.assertIn(
            "dropout_p=self.p_dropout if self.training else 0.0",
            attention,
        )
        self.assertIn("sdpa_mask = sdpa_mask.masked_fill(", attention)
        self.assertIn("TORCH_PROFILE_OUTPUT_DIR", launcher)
        callbacks = (
            ROOT / "stereo_tokenizer" / "modules" / "callbacks.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "self.check_frequency(pl_module.global_step, split)", callbacks
        )
        self.assertNotIn("self.check_frequency(batch_idx)", callbacks)
        model = (ROOT / "stereo_tokenizer" / "model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('on_epoch=not prefix.startswith("train/")', model)
        self.assertNotIn("--causal_in_temporal_transformer", launcher)

    def test_train_launcher_online_gt_is_explicit_and_cache_defaults_off(self) -> None:
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("STEREO_DATA_BACKEND", launcher)
        self.assertIn('--hy_manifest "${HY_MANIFEST}"', launcher)
        self.assertIn('--libero_manifest "${LIBERO_MANIFEST}"', launcher)
        self.assertIn('--umi_manifest "${UMI_MANIFEST}"', launcher)
        self.assertIn("--online_gt_enabled 1", launcher)
        self.assertIn(
            'ONLINE_GT_CACHE_ENABLED="${ONLINE_GT_CACHE_ENABLED:-0}"',
            launcher,
        )
        self.assertIn("--foundation_stereo_valid_iters", launcher)
        self.assertNotIn("--lerobot_episode_manifest", launcher)
        self.assertNotIn("--lerobot_rectification_audit_sha256", launcher)
        train = _training_source("runtime.py")
        self.assertIn("use_distributed_sampler=False", train)
        self.assertNotIn("max_time=", train)

    def test_train_launcher_has_fail_closed_optional_ib_mode(self) -> None:
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        train = _training_source("runtime.py")
        probe = (ROOT / "scripts/stereo/check_ib_collective.py").read_text(
            encoding="utf-8"
        )
        for token in (
            'DISTRIBUTED_MODE="${DISTRIBUTED_MODE:-single}"',
            'NUM_NODES="${NUM_NODES:-1}"',
            "WORLD_SIZE=$((NUM_NODES * GPU_COUNT))",
            "--nproc_per_node",
            "--node_rank",
            "--master_addr",
            "--master_port",
            '--num_nodes "${NUM_NODES}"',
            '--distributed_mode "${DISTRIBUTED_MODE}"',
            "NCCL_IB_DISABLE=0",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA",
            'grep -q "NET/IB"',
        ):
            self.assertIn(token, launcher)
        self.assertIn('choices=("single", "ib")', train)
        self.assertIn("validate_distributed_runtime_args(args)", train)
        self.assertIn("_validate_four_mode_batch_contract(args)", train)
        self.assertIn('backend="nccl"', probe)
        self.assertIn("dist.all_reduce", probe)
        self.assertIn("dist.all_gather_object", probe)

    def test_three_source_training_wires_manifests_da3_and_checkpoint(self) -> None:
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        train = _training_source("runtime.py", "callbacks.py", "checkpoints.py")
        model = (ROOT / "stereo_tokenizer/model.py").read_text(encoding="utf-8")
        online_gt = (ROOT / "stereo_tokenizer/online_gt.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "HY_MANIFEST",
            "LIBERO_MANIFEST",
            "UMI_MANIFEST",
            "UMI_DATASET_ROOT",
            "UMI_RECTIFICATION_AUDIT_SHA256",
            "MODE_UPDATES_PER_EPOCH",
            "DA3_CHECKPOINT_SHA256",
            "NODE_MANIFEST_CONTRACTS",
            "MODE_UPDATE_WEIGHTS",
            "MODE_BATCH_SIZES",
            "MODE_GRAD_ACCUMULATES",
            "MONO_DATASET_WEIGHTS",
            "STEREO_TRAINING_INPUT",
        ):
            self.assertIn(token, launcher)
        self.assertNotIn("UMI_ROOT_ALIASES", launcher)
        self.assertNotIn("UMI_EPISODE_CACHE_CAPACITY", launcher)
        self.assertNotIn("four-mode training is frozen to per-device BS24", launcher)
        self.assertIn(
            'MODE_BATCH_SIZES="${MODE_BATCH_SIZES:-192:40:160:36}"',
            launcher,
        )
        self.assertIn(
            'MODE_GRAD_ACCUMULATES="${MODE_GRAD_ACCUMULATES:-1:1:1:1}"',
            launcher,
        )
        self.assertIn('"val/mixed/total_loss"', train)
        self.assertIn("OnlineDepthAnything3GTCallback", train)
        self.assertIn("mode_occurrences_before", train)
        self.assertIn("mode_for_update", model)
        self.assertNotIn("teacher_rgb_raw", model)
        self.assertIn("da3_images", online_gt)
        self.assertIn("DepthAnything3OnlineTeacher", online_gt)
        self.assertIn("finite_positive_non_padding", launcher)
        self.assertIn(
            '--stereo_training_input "${STEREO_TRAINING_INPUT}"',
            launcher,
        )

    def test_stereo_input_ablation_wrapper_is_serial_and_fail_closed(self) -> None:
        wrapper = (
            ROOT / "scripts/stereo/run_stereo_input_ablation_serial.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("for condition in left_only same_left", wrapper)
        self.assertIn("export LATENT_CHANNELS=48", wrapper)
        self.assertIn('if [[ -e "${ABLATION_OUTPUT_ROOT}" ]]', wrapper)
        self.assertIn('exit "${train_status}"', wrapper)

    def test_tensorrt_backend_is_explicit_and_frozen_to_32_iterations(self):
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        train = _training_source("runtime.py")
        online_gt = (ROOT / "stereo_tokenizer/online_gt.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "FOUNDATION_STEREO_BACKEND",
            "FOUNDATION_STEREO_ENGINE_SHA256",
            "FOUNDATION_STEREO_ENGINE_MANIFEST_SHA256",
            'FOUNDATION_STEREO_VALID_ITERS}" != "32"',
        ):
            self.assertIn(token, launcher)
        for argument in (
            "--foundation_stereo_backend",
            "--foundation_stereo_engine",
            "--foundation_stereo_engine_sha256",
            "--foundation_stereo_engine_manifest",
            "--foundation_stereo_engine_manifest_sha256",
        ):
            self.assertIn(argument, train)
        self.assertIn("execute_async_v3", online_gt)
        self.assertIn("torch.cuda.current_stream", online_gt)
        self.assertIn("set_tensor_address", online_gt)

    def test_tensorrt_comparison_does_not_change_iteration_ablation(self):
        comparison = (
            ROOT / "scripts/stereo/compare_foundation_backends.py"
        ).read_text(encoding="utf-8")
        original = (
            ROOT / "scripts/stereo/compare_online_foundation_teacher.py"
        ).read_text(encoding="utf-8")
        self.assertIn('choices=("equivalence", "tensorrt_benchmark")', comparison)
        self.assertIn("equivalence comparison is limited to 32-64 samples", comparison)
        self.assertIn("backend pilot is frozen to the approved 408 selection", comparison)
        self.assertIn('valid_iters=32', comparison)
        self.assertIn("args.valid_iters != [32, 16, 12]", original)

    def test_tensorrt_manifest_writer_is_fail_closed(self):
        source = (
            ROOT / "scripts/stereo/write_foundation_tensorrt_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn("refusing to overwrite", source)
        self.assertIn("expected_profile", source)
        self.assertIn('"valid_iters": 32', source)
        self.assertIn('"precision": "fp16"', source)
        self.assertIn('"xformers_disabled": True', source)

    def test_online_teacher_comparison_freezes_requested_order(self) -> None:
        source = (
            ROOT / "scripts/stereo/compare_online_foundation_teacher.py"
        ).read_text(encoding="utf-8")
        self.assertIn("args.valid_iters != [32, 16, 12]", source)
        self.assertIn("valid_mask_iou_with_32", source)
        self.assertIn("automatic_numeric_pass", source)
        self.assertIn("if not args.allow_pending_visual_review:", source)
        self.assertIn('"visual_sample_ids": sorted(visual_sample_ids)', source)

if __name__ == "__main__":
    unittest.main()
