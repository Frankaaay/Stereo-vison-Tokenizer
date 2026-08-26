"""Opt-in profiling helpers for the StereoVAE profiling branch."""

from __future__ import annotations

from contextlib import contextmanager

from torch.profiler import record_function


_ENABLED = False


def set_profiling_enabled(enabled: bool) -> None:
    """Enable or disable named profiler regions without changing tensor math."""

    global _ENABLED
    _ENABLED = bool(enabled)


@contextmanager
def profile_region(name: str):
    if not _ENABLED:
        yield
        return
    with record_function(name):
        yield
