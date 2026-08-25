"""Frozen bidirectional FoundationStereo teacher for online disparity targets."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from .profiling import profile_region


CACHE_SCHEMA = "stereo-online-foundation-gt-v2"
TENSORRT_ENGINE_MANIFEST_SCHEMA = "foundation-stereo-tensorrt-engine-v1"
TENSORRT_BINDING_ROLES = ("left", "right", "disparity")


class _UnavailableOpen3D(types.ModuleType):
    """Import-only stub for FoundationStereo's unused point-cloud helpers."""

    def __getattr__(self, name):
        raise RuntimeError(
            "Open3D is unavailable in online FoundationStereo inference; "
            f"attempted to access open3d.{name}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, field):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a full SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error


def _require_git_sha(value, field):
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{field} must be a full Git SHA")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error


def validate_tensorrt_engine_assets(
    engine,
    engine_sha256,
    manifest,
    manifest_sha256,
    checkpoint_sha256,
):
    """Validate immutable TensorRT assets without importing TensorRT."""
    engine = Path(engine).expanduser().resolve()
    manifest = Path(manifest).expanduser().resolve()
    for value, field in (
        (engine_sha256, "TensorRT engine SHA256"),
        (manifest_sha256, "TensorRT engine manifest SHA256"),
        (checkpoint_sha256, "FoundationStereo checkpoint SHA256"),
    ):
        _require_sha256(value, field)
    if not engine.is_file():
        raise FileNotFoundError(engine)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if sha256_file(engine) != engine_sha256:
        raise ValueError("TensorRT engine SHA256 mismatch")
    if sha256_file(manifest) != manifest_sha256:
        raise ValueError("TensorRT engine manifest SHA256 mismatch")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid TensorRT engine manifest JSON") from error
    if payload.get("schema") != TENSORRT_ENGINE_MANIFEST_SCHEMA:
        raise ValueError("unsupported TensorRT engine manifest schema")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("TensorRT manifest artifacts must be an object")
    _require_git_sha(
        artifacts.get("foundation_stereo_repo_sha"),
        "manifest FoundationStereo repo SHA",
    )
    for key in (
        "checkpoint_sha256",
        "config_sha256",
        "onnx_sha256",
        "engine_sha256",
    ):
        _require_sha256(artifacts.get(key), f"manifest {key}")
    if artifacts["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("TensorRT manifest checkpoint SHA256 mismatch")
    if artifacts["engine_sha256"] != engine_sha256:
        raise ValueError("TensorRT manifest engine SHA256 mismatch")

    build = payload.get("build")
    expected_build = {
        "height": 256,
        "width": 256,
        "valid_iters": 32,
        "opset": 16,
        "precision": "fp16",
        "xformers_disabled": True,
        "input_layout": "NCHW",
        "input_range": [0.0, 255.0],
        "output_semantics": "left_positive_disparity_px",
    }
    if not isinstance(build, dict):
        raise ValueError("TensorRT manifest build must be an object")
    for key, expected in expected_build.items():
        if build.get(key) != expected:
            raise ValueError(
                f"TensorRT manifest build.{key} must be {expected!r}"
            )
    profile = build.get("batch_profile")
    if profile != {"min": 1, "opt": 48, "max": 48}:
        raise ValueError("TensorRT manifest batch profile must be 1/48/48")

    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(
        TENSORRT_BINDING_ROLES
    ):
        raise ValueError("TensorRT manifest must declare left/right/disparity")
    expected_bindings = {
        "left": ("input", [-1, 3, 256, 256]),
        "right": ("input", [-1, 3, 256, 256]),
        "disparity": ("output", [-1, 1, 256, 256]),
    }
    names = set()
    for role, (mode, shape) in expected_bindings.items():
        binding = bindings[role]
        if not isinstance(binding, dict):
            raise ValueError(f"TensorRT manifest binding {role} must be an object")
        name = binding.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"TensorRT manifest binding {role} needs a name")
        names.add(name)
        if binding.get("mode") != mode:
            raise ValueError(f"TensorRT manifest binding {role} mode mismatch")
        if binding.get("dtype") not in {"float16", "float32"}:
            raise ValueError(f"TensorRT manifest binding {role} dtype is unsupported")
        if binding.get("shape") != shape:
            raise ValueError(f"TensorRT manifest binding {role} shape mismatch")
    if len(names) != len(TENSORRT_BINDING_ROLES):
        raise ValueError("TensorRT manifest binding names must be unique")

    environment = payload.get("environment")
    for key in ("tensorrt", "cuda", "driver", "gpu"):
        if not isinstance(environment, dict) or not environment.get(key):
            raise ValueError(f"TensorRT manifest environment.{key} is required")
    commands = payload.get("commands")
    for key in ("onnx_export", "trtexec_build"):
        if not isinstance(commands, dict) or not commands.get(key):
            raise ValueError(f"TensorRT manifest commands.{key} is required")
    return payload


class FoundationStereoTensorRTRunner:
    """TensorRT v10 runner using direct PyTorch CUDA tensor bindings."""

    def __init__(
        self,
        engine,
        engine_sha256,
        manifest,
        manifest_sha256,
        checkpoint_sha256,
        *,
        device,
    ):
        self.engine_path = Path(engine).expanduser().resolve()
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT FoundationStereo requires a CUDA device")
        current_device = torch.cuda.current_device()
        device_index = (
            current_device if self.device.index is None else self.device.index
        )
        if device_index != current_device:
            raise RuntimeError(
                "TensorRT runner must be created on the current rank CUDA device"
            )
        self.device = torch.device("cuda", device_index)
        self.manifest = validate_tensorrt_engine_assets(
            self.engine_path,
            engine_sha256,
            self.manifest_path,
            manifest_sha256,
            checkpoint_sha256,
        )
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError(
                "TensorRT backend requires the tensorrt Python package"
            ) from error
        for attribute in (
            "Runtime",
            "Logger",
            "TensorIOMode",
            "TensorLocation",
            "TensorFormat",
        ):
            if not hasattr(trt, attribute):
                raise RuntimeError("TensorRT v10 Python API is required")
        runtime_environment = {
            "tensorrt": str(getattr(trt, "__version__", "")),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(self.device),
        }
        for key, actual in runtime_environment.items():
            if self.manifest["environment"][key] != actual:
                raise ValueError(
                    f"TensorRT runtime {key} {actual!r} does not match manifest"
                )
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self._engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        if not hasattr(self._engine, "num_io_tensors"):
            raise RuntimeError("TensorRT engine does not expose the v10 IO API")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self._bindings = self.manifest["bindings"]
        self._profile = self.manifest["build"]["batch_profile"]
        self._validate_engine_contract()

    def _dtype(self, name):
        dtype = self._engine.get_tensor_dtype(name)
        if dtype == self._trt.float16:
            return torch.float16
        if dtype == self._trt.float32:
            return torch.float32
        raise ValueError(f"TensorRT binding {name} has unsupported dtype {dtype}")

    def _validate_engine_contract(self):
        if getattr(self._engine, "num_optimization_profiles", 0) != 1:
            raise ValueError("TensorRT engine must contain exactly one profile")
        actual_names = {
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        }
        expected_names = {
            binding["name"] for binding in self._bindings.values()
        }
        if actual_names != expected_names:
            raise ValueError("TensorRT engine IO names do not match manifest")
        for role, binding in self._bindings.items():
            name = binding["name"]
            expected_mode = (
                self._trt.TensorIOMode.INPUT
                if binding["mode"] == "input"
                else self._trt.TensorIOMode.OUTPUT
            )
            if self._engine.get_tensor_mode(name) != expected_mode:
                raise ValueError(f"TensorRT engine binding {role} mode mismatch")
            if (
                self._engine.get_tensor_location(name)
                != self._trt.TensorLocation.DEVICE
            ):
                raise ValueError(f"TensorRT engine binding {role} is not device IO")
            if (
                self._engine.get_tensor_format(name)
                != self._trt.TensorFormat.LINEAR
            ):
                raise ValueError(f"TensorRT engine binding {role} is not linear")
            if self._dtype(name) != {
                "float16": torch.float16,
                "float32": torch.float32,
            }[binding["dtype"]]:
                raise ValueError(f"TensorRT engine binding {role} dtype mismatch")
            actual_shape = tuple(
                int(value) for value in self._engine.get_tensor_shape(name)
            )
            if actual_shape != tuple(binding["shape"]):
                raise ValueError(f"TensorRT engine binding {role} shape mismatch")
            if binding["mode"] == "input":
                profile_shapes = self._engine.get_tensor_profile_shape(name, 0)
                expected = tuple(
                    (batch, 3, 256, 256)
                    for batch in (
                        self._profile["min"],
                        self._profile["opt"],
                        self._profile["max"],
                    )
                )
                actual = tuple(
                    tuple(int(value) for value in shape)
                    for shape in profile_shapes
                )
                if actual != expected:
                    raise ValueError(
                        f"TensorRT engine binding {role} profile mismatch"
                    )

    def infer(self, left, right):
        if left.shape != right.shape:
            raise ValueError("TensorRT left/right input shapes differ")
        if left.ndim != 4 or tuple(left.shape[1:]) != (3, 256, 256):
            raise ValueError("TensorRT inputs must be [N,3,256,256]")
        if left.device != self.device or right.device != self.device:
            raise ValueError("TensorRT inputs must be on the configured CUDA device")
        batch = int(left.shape[0])
        if not self._profile["min"] <= batch <= self._profile["max"]:
            raise ValueError("TensorRT input batch is outside the engine profile")

        left_name = self._bindings["left"]["name"]
        right_name = self._bindings["right"]["name"]
        output_name = self._bindings["disparity"]["name"]
        left = left.to(dtype=self._dtype(left_name)).contiguous()
        right = right.to(dtype=self._dtype(right_name)).contiguous()
        if not self._context.set_input_shape(left_name, tuple(left.shape)):
            raise RuntimeError("TensorRT rejected the left input shape")
        if not self._context.set_input_shape(right_name, tuple(right.shape)):
            raise RuntimeError("TensorRT rejected the right input shape")
        missing_shapes = self._context.infer_shapes()
        if missing_shapes:
            raise RuntimeError(
                "TensorRT shape inference is incomplete for "
                + ", ".join(missing_shapes)
            )
        output_shape = tuple(
            int(value) for value in self._context.get_tensor_shape(output_name)
        )
        expected_output_shape = (batch, 1, 256, 256)
        if output_shape != expected_output_shape:
            raise RuntimeError(
                f"TensorRT resolved output shape {output_shape}, "
                f"expected {expected_output_shape}"
            )
        output = torch.empty(
            output_shape,
            device=self.device,
            dtype=self._dtype(output_name),
        )
        for name, tensor in (
            (left_name, left),
            (right_name, right),
            (output_name, output),
        ):
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected binding address for {name}")
        stream = torch.cuda.current_stream(self.device)
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT FoundationStereo execution failed")
        left.record_stream(stream)
        right.record_stream(stream)
        return output


class FoundationStereoOnlineTeacher:
    """Non-checkpointed frozen FoundationStereo inference wrapper."""

    def __init__(
        self,
        repo,
        checkpoint,
        checkpoint_sha256,
        *,
        device,
        valid_iters,
        pair_microbatch,
        backend="pytorch",
        engine=None,
        engine_sha256=None,
        engine_manifest=None,
        engine_manifest_sha256=None,
    ):
        self.backend = str(backend)
        if self.backend not in {"pytorch", "tensorrt"}:
            raise ValueError(f"unsupported FoundationStereo backend {self.backend}")
        self.repo = (
            Path(repo).expanduser().resolve() if repo is not None else None
        )
        self.checkpoint = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else None
        )
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = torch.device(device)
        self.valid_iters = int(valid_iters)
        self.pair_microbatch = int(pair_microbatch)
        if self.pair_microbatch < 1:
            raise ValueError("FoundationStereo pair microbatch must be positive")
        self.model = None
        self.config = None
        self.runner = None
        if self.backend == "pytorch":
            if self.valid_iters not in {12, 16, 32}:
                raise ValueError(
                    "online FoundationStereo iterations must be 12, 16, or 32"
                )
            if (
                self.repo is None
                or self.checkpoint is None
                or not self.repo.is_dir()
                or not self.checkpoint.is_file()
            ):
                raise FileNotFoundError("FoundationStereo repo/checkpoint is missing")
            if sha256_file(self.checkpoint) != checkpoint_sha256:
                raise ValueError("FoundationStereo checkpoint SHA256 mismatch")
            self.model, self.config = self._load_model()
        else:
            if self.valid_iters != 32:
                raise ValueError("TensorRT FoundationStereo is frozen to 32 iterations")
            if self.pair_microbatch > 48:
                raise ValueError(
                    "TensorRT pair microbatch exceeds the frozen max batch 48"
                )
            self.runner = FoundationStereoTensorRTRunner(
                engine,
                engine_sha256,
                engine_manifest,
                engine_manifest_sha256,
                checkpoint_sha256,
                device=self.device,
            )

    def _load_model(self):
        try:
            import timm
            from omegaconf import OmegaConf
        except ImportError as error:
            raise RuntimeError(
                "online FoundationStereo requires timm and omegaconf"
            ) from error

        original_open3d = sys.modules.get("open3d")
        sys.modules["open3d"] = _UnavailableOpen3D("open3d")
        sys.path.insert(0, str(self.repo))
        try:
            from core.foundation_stereo import FoundationStereo
        finally:
            if sys.path[0] == str(self.repo):
                sys.path.pop(0)
            if original_open3d is None:
                sys.modules.pop("open3d", None)
            else:
                sys.modules["open3d"] = original_open3d

        config = OmegaConf.load(self.checkpoint.parent / "cfg.yaml")
        if "vit_size" not in config:
            config["vit_size"] = "vitl"
        original_create_model = timm.create_model
        original_hub_load = torch.hub.load

        def offline_create_model(model_name, *args, **kwargs):
            if model_name != "edgenext_small":
                raise RuntimeError(f"unexpected timm model request: {model_name}")
            kwargs["pretrained"] = False
            return original_create_model(model_name, *args, **kwargs)

        def offline_hub_load(repo_or_dir, model_name, *args, **kwargs):
            if repo_or_dir != "facebookresearch/dinov2":
                raise RuntimeError(f"unexpected torch.hub request: {repo_or_dir}")
            kwargs["source"] = "local"
            kwargs["pretrained"] = False
            return original_hub_load(
                str(self.repo / "dinov2"), model_name, *args, **kwargs
            )

        timm.create_model = offline_create_model
        torch.hub.load = offline_hub_load
        try:
            model = FoundationStereo(config)
        finally:
            timm.create_model = original_create_model
            torch.hub.load = original_hub_load
        # The full FoundationStereo training payload contains NumPy metadata.
        # Its bytes are trusted only after the frozen SHA256 check in __init__.
        payload = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False)
        model.to(self.device)
        model.eval()
        return model, config

    def _infer_microbatch(self, left, right):
        if self.backend == "tensorrt":
            with torch.inference_mode():
                disparity_left = self.runner.infer(left, right)
                disparity_right = self.runner.infer(
                    torch.flip(right, dims=[3]),
                    torch.flip(left, dims=[3]),
                )
                disparity_right = torch.flip(disparity_right, dims=[3])
            return disparity_left.float(), disparity_right.float()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            disparity_left = self.model.forward(
                left, right, iters=self.valid_iters, test_mode=True
            )
            disparity_right = self.model.forward(
                torch.flip(right, dims=[3]),
                torch.flip(left, dims=[3]),
                iters=self.valid_iters,
                test_mode=True,
            )
            disparity_right = torch.flip(disparity_right, dims=[3])
        return disparity_left.float(), disparity_right.float()

    def infer(self, video):
        """Infer bidirectional disparity for [B,V,E,C,T,H,W] RGB in [-.5,.5]."""
        if video.ndim != 7 or video.shape[1:5] != (3, 2, 3, 4):
            raise ValueError(
                "online teacher expects video [B,3,2,3,4,H,W]"
            )
        left = video[:, :, 0].permute(0, 1, 3, 2, 4, 5).contiguous()
        right = video[:, :, 1].permute(0, 1, 3, 2, 4, 5).contiguous()
        batch, views, frames, channels, height, width = left.shape
        left = (left.reshape(-1, channels, height, width) + 0.5) * 255.0
        right = (right.reshape(-1, channels, height, width) + 0.5) * 255.0
        left_outputs = []
        right_outputs = []
        for start in range(0, left.shape[0], self.pair_microbatch):
            stop = start + self.pair_microbatch
            disparity_left, disparity_right = self._infer_microbatch(
                left[start:stop], right[start:stop]
            )
            left_outputs.append(disparity_left)
            right_outputs.append(disparity_right)
        disparity_left = torch.cat(left_outputs)
        disparity_right = torch.cat(right_outputs)
        residual, base_valid = self.lr_consistency(
            disparity_left, disparity_right
        )
        shape = (batch, views, frames, 1, height, width)
        disparity_left = disparity_left.reshape(shape).permute(0, 1, 3, 2, 4, 5)
        residual = residual.reshape(shape).permute(0, 1, 3, 2, 4, 5)
        base_valid = base_valid.reshape(shape).permute(0, 1, 3, 2, 4, 5)
        return disparity_left, residual, base_valid

    @staticmethod
    def lr_consistency(disparity_left, disparity_right):
        if disparity_left.shape != disparity_right.shape:
            raise ValueError("left/right teacher disparity shapes differ")
        if disparity_left.ndim != 4 or disparity_left.shape[1] != 1:
            raise ValueError("teacher disparity must be [N,1,H,W]")
        _, _, height, width = disparity_left.shape
        x = torch.arange(
            width, device=disparity_left.device, dtype=disparity_left.dtype
        ).view(1, 1, width)
        y = torch.arange(
            height, device=disparity_left.device, dtype=disparity_left.dtype
        ).view(1, height, 1)
        x_right = x - disparity_left[:, 0]
        grid_x = 2.0 * x_right / max(width - 1, 1) - 1.0
        grid_y = 2.0 * y.expand_as(grid_x) / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        sampled_right = F.grid_sample(
            disparity_right,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        finite = torch.isfinite(disparity_left) & torch.isfinite(sampled_right)
        valid = (
            finite
            & (disparity_left > 0)
            & (sampled_right > 0)
            & (x_right[:, None] >= 0)
            & (x_right[:, None] <= width - 1)
        )
        content = torch.zeros(
            (1, 1, height, width),
            device=disparity_left.device,
            dtype=torch.bool,
        )
        content[:, :, 32:224] = True
        valid &= content
        residual = torch.full_like(disparity_left, float("nan"))
        difference = torch.abs(disparity_left - sampled_right)
        residual = torch.where(valid, difference, residual)
        return residual, valid


class OnlineFoundationGTCallback(Callback):
    """Generate disparity targets after device transfer and before each step."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.teacher = None
        self.cache_enabled = bool(args.online_gt_cache_enabled)
        self.cache_root = (
            Path(args.online_gt_cache_root).expanduser().resolve()
            if args.online_gt_cache_root
            else None
        )
        if self.cache_enabled and self.cache_root is None:
            raise ValueError("online GT cache requires --online_gt_cache_root")
        self.backend = args.foundation_stereo_backend
        cache_provenance = {
            "backend": self.backend,
            "checkpoint_sha256": args.foundation_stereo_checkpoint_sha256,
            "valid_iters": args.foundation_stereo_valid_iters,
            "engine_sha256": getattr(
                args, "foundation_stereo_engine_sha256", None
            ),
            "engine_manifest_sha256": getattr(
                args, "foundation_stereo_engine_manifest_sha256", None
            ),
        }
        self.cache_namespace = hashlib.sha256(
            json.dumps(
                cache_provenance, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    @property
    def state_key(self):
        return (
            f"{self.__class__.__qualname__}:"
            f"{self.backend}:"
            f"{self.args.foundation_stereo_checkpoint_sha256}:"
            f"{self.args.foundation_stereo_valid_iters}:"
            f"{getattr(self.args, 'foundation_stereo_engine_sha256', None)}:"
            f"{getattr(self.args, 'foundation_stereo_engine_manifest_sha256', None)}"
        )

    def state_dict(self):
        return {}

    def setup(self, trainer, pl_module, stage=None):
        if stage not in (None, "fit") or self.teacher is not None:
            return
        self.teacher = FoundationStereoOnlineTeacher(
            self.args.foundation_stereo_repo,
            self.args.foundation_stereo_checkpoint,
            self.args.foundation_stereo_checkpoint_sha256,
            device=trainer.strategy.root_device,
            valid_iters=self.args.foundation_stereo_valid_iters,
            pair_microbatch=self.args.foundation_stereo_pair_microbatch,
            backend=self.backend,
            engine=getattr(self.args, "foundation_stereo_engine", None),
            engine_sha256=getattr(
                self.args, "foundation_stereo_engine_sha256", None
            ),
            engine_manifest=getattr(
                self.args, "foundation_stereo_engine_manifest", None
            ),
            engine_manifest_sha256=getattr(
                self.args, "foundation_stereo_engine_manifest_sha256", None
            ),
        )

    def teardown(self, trainer, pl_module, stage=None):
        self.teacher = None

    def _cache_path(self, sample_id):
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        return (
            self.cache_root
            / self.backend
            / self.cache_namespace
            / digest[:2]
            / f"{digest}.npz"
        )

    def _cache_metadata(self, sample_id, contract_sha256):
        return {
            "schema": CACHE_SCHEMA,
            "sample_id": sample_id,
            "contract_sha256": contract_sha256,
            "backend": self.backend,
            "checkpoint_sha256": self.args.foundation_stereo_checkpoint_sha256,
            "valid_iters": self.args.foundation_stereo_valid_iters,
            "engine_sha256": getattr(
                self.args, "foundation_stereo_engine_sha256", None
            ),
            "engine_manifest_sha256": getattr(
                self.args, "foundation_stereo_engine_manifest_sha256", None
            ),
            "bidirectional": True,
            "lr_consistency": True,
        }

    def _read_cache(self, sample_id, contract_sha256):
        if not self.cache_enabled:
            return None
        path = self._cache_path(sample_id)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != {"disparity", "valid_mask", "metadata_json"}:
                raise ValueError(f"{path}: invalid online GT cache keys")
            metadata = json.loads(str(cache["metadata_json"]))
            if metadata != self._cache_metadata(sample_id, contract_sha256):
                raise ValueError(f"{path}: online GT cache metadata mismatch")
            if cache["disparity"].dtype != np.float32:
                raise ValueError(f"{path}: online GT cache disparity must be float32")
            if cache["valid_mask"].dtype != np.bool_:
                raise ValueError(f"{path}: online GT cache mask must be bool")
            disparity = cache["disparity"].astype(np.float32)
            valid = cache["valid_mask"].astype(np.bool_)
        if disparity.shape != (3, 1, 4, 256, 256) or valid.shape != disparity.shape:
            raise ValueError(f"{path}: invalid online GT cache shape")
        return disparity, valid

    def _write_cache(self, sample_id, contract_sha256, disparity, valid):
        path = self._cache_path(sample_id)
        if path.exists():
            cached = self._read_cache(sample_id, contract_sha256)
            if cached is None:
                raise RuntimeError(f"failed to validate existing cache {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".partial-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        metadata = np.array(
            json.dumps(
                self._cache_metadata(sample_id, contract_sha256),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    disparity=disparity.astype(np.float32),
                    valid_mask=valid.astype(np.bool_),
                    metadata_json=metadata,
                )
            try:
                os.link(temporary, path)
            except FileExistsError:
                cached = self._read_cache(sample_id, contract_sha256)
                if cached is None:
                    raise RuntimeError(f"failed to validate concurrent cache {path}")
            temporary.unlink()
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

    def _generate(self, batch, pl_module):
        if self.teacher is None:
            raise RuntimeError("online FoundationStereo teacher is not initialized")
        if not self.cache_enabled:
            disparity, residual, base_valid = self.teacher.infer(batch["video"])
            threshold = torch.maximum(
                residual.new_tensor(self.args.stereo_lr_error_abs_threshold_px),
                self.args.stereo_lr_error_relative_threshold * disparity,
            )
            batch["disparity"] = disparity
            batch["valid_mask"] = (
                base_valid
                & torch.isfinite(disparity)
                & torch.isfinite(residual)
                & (disparity >= self.args.stereo_disparity_min_px)
                & (disparity <= self.args.stereo_disparity_max_px)
                & (residual <= threshold)
            )
            return int(batch["video"].shape[0])

        sample_ids = list(batch["sample_id"])
        contract_hashes = list(batch["contract_sha256"])
        results = [None] * len(sample_ids)
        missing = []
        for index, (sample_id, contract_hash) in enumerate(
            zip(sample_ids, contract_hashes)
        ):
            cached = self._read_cache(sample_id, contract_hash)
            if cached is None:
                missing.append(index)
            else:
                results[index] = cached

        if missing:
            missing_tensor = torch.tensor(
                missing, device=batch["video"].device, dtype=torch.long
            )
            video = batch["video"].index_select(0, missing_tensor)
            disparity, residual, base_valid = self.teacher.infer(video)
            threshold = torch.maximum(
                residual.new_tensor(self.args.stereo_lr_error_abs_threshold_px),
                self.args.stereo_lr_error_relative_threshold * disparity,
            )
            valid = (
                base_valid
                & torch.isfinite(disparity)
                & torch.isfinite(residual)
                & (disparity >= self.args.stereo_disparity_min_px)
                & (disparity <= self.args.stereo_disparity_max_px)
                & (residual <= threshold)
            )
            for output_index, batch_index in enumerate(missing):
                item = (
                    disparity[output_index].detach().cpu().numpy(),
                    valid[output_index].detach().cpu().numpy(),
                )
                results[batch_index] = item
                if self.cache_enabled:
                    self._write_cache(
                        sample_ids[batch_index],
                        contract_hashes[batch_index],
                        item[0],
                        item[1],
                    )

        disparity = torch.from_numpy(np.stack([item[0] for item in results])).to(
            device=batch["video"].device, dtype=torch.float32
        )
        valid = torch.from_numpy(np.stack([item[1] for item in results])).to(
            device=batch["video"].device, dtype=torch.bool
        )
        batch["disparity"] = disparity
        batch["valid_mask"] = valid
        return len(missing)

    def _prepare_batch(self, trainer, pl_module, batch, prefix):
        started = time.perf_counter()
        with profile_region("stereo/online_gt/foundation_teacher"):
            generated_count = self._generate(batch, pl_module)
        torch.cuda.synchronize(batch["video"].device)
        seconds = time.perf_counter() - started
        pl_module.log(
            f"{prefix}/online_gt_seconds",
            seconds,
            on_step=prefix == "train",
            on_epoch=prefix != "train",
            sync_dist=True,
        )
        pl_module.log(
            f"{prefix}/online_gt_generated_samples",
            float(generated_count),
            on_step=prefix == "train",
            on_epoch=prefix != "train",
            sync_dist=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._prepare_batch(trainer, pl_module, batch, "train")

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._prepare_batch(trainer, pl_module, batch, "val")
