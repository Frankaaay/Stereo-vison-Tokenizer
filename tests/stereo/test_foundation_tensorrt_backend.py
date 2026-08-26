import hashlib
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from stereo_tokenizer.online_gt import (
    FoundationStereoOnlineTeacher,
    FoundationStereoTensorRTRunner,
    OnlineFoundationGTCallback,
    TENSORRT_ENGINE_MANIFEST_SCHEMA,
    validate_tensorrt_engine_assets,
)


CHECKPOINT_SHA256 = "6" * 64
REPO_SHA = "7" * 40


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload(engine_sha256):
    return {
        "schema": TENSORRT_ENGINE_MANIFEST_SCHEMA,
        "artifacts": {
            "foundation_stereo_repo_sha": REPO_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": "8" * 64,
            "onnx_sha256": "9" * 64,
            "engine_sha256": engine_sha256,
        },
        "build": {
            "height": 256,
            "width": 256,
            "valid_iters": 32,
            "opset": 16,
            "precision": "fp16",
            "xformers_disabled": True,
            "input_layout": "NCHW",
            "input_range": [0.0, 255.0],
            "output_semantics": "left_positive_disparity_px",
            "batch_profile": {"min": 1, "opt": 48, "max": 48},
        },
        "bindings": {
            "left": {
                "name": "left",
                "mode": "input",
                "dtype": "float32",
                "shape": [-1, 3, 256, 256],
            },
            "right": {
                "name": "right",
                "mode": "input",
                "dtype": "float32",
                "shape": [-1, 3, 256, 256],
            },
            "disparity": {
                "name": "disparity",
                "mode": "output",
                "dtype": "float32",
                "shape": [-1, 1, 256, 256],
            },
        },
        "environment": {
            "tensorrt": "10.test",
            "cuda": "12.test",
            "driver": "test",
            "gpu": "H200",
        },
        "commands": {
            "onnx_export": "python export_onnx.py ...",
            "trtexec_build": "trtexec --fp16 ...",
        },
    }


def write_assets(root):
    engine = root / "foundation_stereo.plan"
    engine.write_bytes(b"test TensorRT engine")
    engine_sha = sha256(engine)
    manifest = root / "engine_manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload(engine_sha), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return engine, engine_sha, manifest, sha256(manifest)


class _FakeContext:
    pass


class _FakeEngine:
    num_io_tensors = 3
    num_optimization_profiles = 1

    def __init__(self, trt):
        self.trt = trt

    def create_execution_context(self):
        return _FakeContext()

    def get_tensor_name(self, index):
        return ("left", "right", "disparity")[index]

    def get_tensor_mode(self, name):
        return (
            self.trt.TensorIOMode.OUTPUT
            if name == "disparity"
            else self.trt.TensorIOMode.INPUT
        )

    def get_tensor_dtype(self, name):
        return self.trt.float32

    def get_tensor_location(self, name):
        return self.trt.TensorLocation.DEVICE

    def get_tensor_format(self, name):
        return self.trt.TensorFormat.LINEAR

    def get_tensor_shape(self, name):
        return (-1, 1, 256, 256) if name == "disparity" else (-1, 3, 256, 256)

    def get_tensor_profile_shape(self, name, profile):
        return (
            (1, 3, 256, 256),
            (48, 3, 256, 256),
            (48, 3, 256, 256),
        )


def fake_tensorrt_module():
    module = types.ModuleType("tensorrt")
    module.__version__ = "10.test"
    module.float16 = object()
    module.float32 = object()
    module.TensorIOMode = SimpleNamespace(INPUT="input", OUTPUT="output")
    module.TensorLocation = SimpleNamespace(DEVICE="device")
    module.TensorFormat = SimpleNamespace(LINEAR="linear")

    class Logger:
        ERROR = "error"

        def __init__(self, level):
            self.level = level

    class Runtime:
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, payload):
            return _FakeEngine(module)

    module.Logger = Logger
    module.Runtime = Runtime
    return module


class _RecordingRunner:
    def __init__(self):
        self.calls = []

    def infer(self, left, right):
        self.calls.append((left.clone(), right.clone()))
        return (left[:, :1] - right[:, :1]).to(torch.float16)


class FoundationTensorRTBackendTest(unittest.TestCase):
    def test_assets_validate_without_importing_tensorrt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, engine_sha, manifest, manifest_sha = write_assets(root)
            previous = sys.modules.pop("tensorrt", None)
            try:
                payload = validate_tensorrt_engine_assets(
                    engine,
                    engine_sha,
                    manifest,
                    manifest_sha,
                    CHECKPOINT_SHA256,
                )
            finally:
                if previous is not None:
                    sys.modules["tensorrt"] = previous
            self.assertEqual(payload["build"]["valid_iters"], 32)

    def test_assets_fail_closed_on_hash_contract_and_profile_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, engine_sha, manifest, manifest_sha = write_assets(root)
            with self.assertRaisesRegex(ValueError, "engine SHA256 mismatch"):
                validate_tensorrt_engine_assets(
                    engine,
                    "0" * 64,
                    manifest,
                    manifest_sha,
                    CHECKPOINT_SHA256,
                )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["build"]["batch_profile"]["max"] = 49
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch profile"):
                validate_tensorrt_engine_assets(
                    engine,
                    engine_sha,
                    manifest,
                    sha256(manifest),
                    CHECKPOINT_SHA256,
                )

    def test_each_runner_owns_a_distinct_execution_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, engine_sha, manifest, manifest_sha = write_assets(root)
            fake_trt = fake_tensorrt_module()
            with mock.patch.dict(sys.modules, {"tensorrt": fake_trt}), mock.patch(
                "torch.cuda.current_device", return_value=0
            ), mock.patch(
                "torch.cuda.get_device_name", return_value="H200"
            ), mock.patch.object(torch.version, "cuda", "12.test"):
                first = FoundationStereoTensorRTRunner(
                    engine,
                    engine_sha,
                    manifest,
                    manifest_sha,
                    CHECKPOINT_SHA256,
                    device="cuda:0",
                )
                second = FoundationStereoTensorRTRunner(
                    engine,
                    engine_sha,
                    manifest,
                    manifest_sha,
                    CHECKPOINT_SHA256,
                    device="cuda:0",
                )
            self.assertIsNot(first._context, second._context)

    def test_bidirectional_backend_preserves_batch_shape_dtype_and_device(self):
        for batch in (1, 36, 48):
            runner = _RecordingRunner()
            teacher = FoundationStereoOnlineTeacher.__new__(
                FoundationStereoOnlineTeacher
            )
            teacher.backend = "tensorrt"
            teacher.runner = runner
            left = torch.arange(batch * 3 * 2 * 4, dtype=torch.float32).reshape(
                batch, 3, 2, 4
            )
            right = left + 1
            disparity_left, disparity_right = teacher._infer_microbatch(left, right)
            self.assertEqual(disparity_left.shape, (batch, 1, 2, 4))
            self.assertEqual(disparity_right.shape, (batch, 1, 2, 4))
            self.assertEqual(disparity_left.dtype, torch.float32)
            self.assertEqual(disparity_left.device, left.device)
            self.assertEqual(len(runner.calls), 2)
            torch.testing.assert_close(
                runner.calls[1][0], torch.flip(right, dims=[3])
            )
            torch.testing.assert_close(
                runner.calls[1][1], torch.flip(left, dims=[3])
            )

    def test_cache_namespace_and_state_key_isolate_backends(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def args(backend):
                return SimpleNamespace(
                    online_gt_cache_enabled=1,
                    online_gt_cache_root=str(root),
                    foundation_stereo_backend=backend,
                    foundation_stereo_checkpoint_sha256=CHECKPOINT_SHA256,
                    foundation_stereo_valid_iters=32,
                    foundation_stereo_engine_sha256=(
                        "a" * 64 if backend == "tensorrt" else None
                    ),
                    foundation_stereo_engine_manifest_sha256=(
                        "b" * 64 if backend == "tensorrt" else None
                    ),
                )

            pytorch = OnlineFoundationGTCallback(args("pytorch"))
            tensorrt = OnlineFoundationGTCallback(args("tensorrt"))
            self.assertNotEqual(
                pytorch._cache_path("sample", "four_frame"),
                tensorrt._cache_path("sample", "four_frame"),
            )
            self.assertNotEqual(pytorch.state_key, tensorrt.state_key)
            metadata = tensorrt._cache_metadata(
                "sample", "c" * 64, "four_frame"
            )
            self.assertEqual(metadata["backend"], "tensorrt")
            self.assertEqual(metadata["engine_sha256"], "a" * 64)
            self.assertEqual(metadata["target_representation"], "pixel_disparity_px")
            self.assertEqual(metadata["tensor_shape"], [3, 1, 4, 256, 256])

    def test_runner_source_has_no_cpu_or_numpy_data_path(self):
        source = inspect.getsource(FoundationStereoTensorRTRunner)
        self.assertNotIn(".cpu(", source)
        self.assertNotIn(".numpy(", source)
        self.assertNotIn("np.", source)
        self.assertIn("torch.cuda.current_stream", source)
        self.assertIn("set_tensor_address", source)
        self.assertIn("execute_async_v3", source)


if __name__ == "__main__":
    unittest.main()
