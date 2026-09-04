"""Stable CLI for the frozen Stereo Tokenizer Stage A evaluation."""

from __future__ import annotations

import sys

from .stage_a.benchmark import _benchmark_command
from .stage_a.quality import _run_command
from .stage_a.report import _report_command
from .stage_a.selection import _preflight_command, _selection_command


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("expected one of: selection, preflight, run, benchmark, report")
    command, argv = sys.argv[1], sys.argv[2:]
    commands = {
        "selection": _selection_command,
        "preflight": _preflight_command,
        "run": _run_command,
        "benchmark": _benchmark_command,
        "report": _report_command,
    }
    if command not in commands:
        raise SystemExit(f"unknown command {command!r}")
    commands[command](argv)


if __name__ == "__main__":
    main()
