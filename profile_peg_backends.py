from __future__ import annotations

import argparse
import json
import statistics
from itertools import product

import torch

from stereo_tokenizer.modules.attention import PEG


BACKENDS = (
    "conv3d_contiguous",
    "conv3d_channels_last_3d",
    "conv2d_t1_slice",
)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().cpu())


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    max_abs = _max_abs(left, right)
    reference_max_abs = float(right.float().abs().max().cpu())
    return {
        "max_abs": max_abs,
        "reference_max_abs": reference_max_abs,
        "max_abs_over_reference_max_abs": (
            max_abs / reference_max_abs if reference_max_abs else 0.0
        ),
    }


def _run_once(
    module: PEG,
    source: torch.Tensor,
    output_grad: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> dict[str, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    value = source.detach().clone().requires_grad_(True)
    output = module(value, shape=shape)
    output.backward(output_grad)
    return {
        "output": output.detach().clone(),
        "input_grad": value.grad.detach().clone(),
        "weight_grad": module.dsconv.weight.grad.detach().clone(),
        "bias_grad": module.dsconv.bias.grad.detach().clone(),
    }


def _benchmark(
    module: PEG,
    source: torch.Tensor,
    output_grad: torch.Tensor,
    shape: tuple[int, int, int, int],
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    timings = []
    for index in range(warmup + iterations):
        module.zero_grad(set_to_none=True)
        value = source.detach().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = module(value, shape=shape)
        output.backward(output_grad)
        end.record()
        end.synchronize()
        if index >= warmup:
            timings.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def _adam_step(module: PEG, source, output_grad, shape) -> dict[str, torch.Tensor]:
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    _run_once(module, source, output_grad, shape)
    optimizer.step()
    state = optimizer.state[module.dsconv.weight]
    return {
        "weight": module.dsconv.weight.detach().clone(),
        "bias": module.dsconv.bias.detach().clone(),
        "exp_avg": state["exp_avg"].detach().clone(),
        "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    device = torch.device("cuda")
    results = []

    for dtype, batch in product(
        (torch.bfloat16, torch.float32), (192, 24)
    ):
        shape = (batch, 1, 16, 16)
        source = torch.randn(batch, 256, 512, device=device, dtype=dtype)
        output_grad = torch.randn_like(source)
        template = PEG(512, causal=True).to(device=device, dtype=dtype)
        initial_state = {
            key: value.detach().clone()
            for key, value in template.state_dict().items()
        }

        modules = {}
        observations = {}
        adam_states = {}
        for backend in BACKENDS:
            module = PEG(512, causal=True).to(device=device, dtype=dtype)
            module.load_state_dict(initial_state)
            module.set_profile_backend(backend)
            modules[backend] = module
            observations[backend] = _run_once(
                module, source, output_grad, shape
            )

            adam_module = PEG(512, causal=True).to(device=device, dtype=dtype)
            adam_module.load_state_dict(initial_state)
            adam_module.set_profile_backend(backend)
            adam_states[backend] = _adam_step(
                adam_module, source, output_grad, shape
            )

        reference = observations[BACKENDS[0]]
        reference_adam = adam_states[BACKENDS[0]]
        for backend in BACKENDS:
            observation = observations[backend]
            inactive = observation["weight_grad"][:, :, :2]
            results.append(
                {
                    "batch": batch,
                    "dtype": str(dtype),
                    "backend": backend,
                    "forward_backward": _benchmark(
                        modules[backend],
                        source,
                        output_grad,
                        shape,
                        args.warmup,
                        args.iterations,
                    ),
                    "comparison_vs_contiguous": {
                        key: _comparison(observation[key], reference[key])
                        for key in observation
                    },
                    "adam_comparison_vs_contiguous": {
                        key: _comparison(adam_states[backend][key], reference_adam[key])
                        for key in reference_adam
                    },
                    "inactive_temporal_weight_grad_max_abs": float(
                        inactive.float().abs().max().cpu()
                    ),
                }
            )

    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
