"""Command line for the standalone Reservoir Pong experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import calibrate_run
from .evaluation_run import run_evaluation_stage
from .preflight import run_preflight
from .records import ExperimentConfig, create_run_directory
from .training import (
    run_pilot,
    run_training_stage,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "experiments" / "pong"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reservoir-pong")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="artifact root (default: experiments/pong)")
    parser.add_argument("--run-dir", type=Path,
                        help="existing run directory; required for all but 'all' and new 'calibrate'")
    parser.add_argument("--name", default="reservoir-pong", help="run name for a new experiment")
    parser.add_argument("--config", type=Path,
                        help="JSON settings for a new run; changing settings creates a new run directory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "preflight", "pilot", "train", "evaluate",
                 "serve", "all"):
        command = sub.add_parser(name)
        if name == "calibrate":
            command.add_argument("--repeats", type=int)
        if name == "preflight":
            command.add_argument("--counts", type=int, nargs=3, default=(120, 40, 80),
                                 metavar=("TRAIN", "VALIDATION", "TEST"))
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
        if name == "evaluate":
            command.add_argument("--test-seed", type=int,
                                 help="fixed seed for an independent held-out block")
        if name == "all":
            command.add_argument("--calibration-repeats", type=int)
            command.add_argument("--skip-pilot", action="store_true")
            command.add_argument("--skip-final", action="store_true")
            command.add_argument("--preflight-counts", type=int, nargs=3,
                                 default=(120, 40, 80))
    return parser


def _latest_run(root: Path) -> Path:
    choices = sorted(path for path in Path(root).glob("*") if path.is_dir())
    if not choices:
        raise FileNotFoundError(f"no experiment runs exist under {root}")
    return choices[-1]


def _resolve_run(args) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    if args.command in {"all", "calibrate"}:
        config = (ExperimentConfig.from_dict(json.loads(args.config.read_text()))
                  if args.config else ExperimentConfig())
        return create_run_directory(args.root, args.name, config)
    return _latest_run(args.root)


def _load_config(run_dir: Path) -> ExperimentConfig:
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"run configuration not found: {path}")
    return ExperimentConfig.from_dict(json.loads(path.read_text()))


def _gate_passed(run_dir: Path, name: str) -> bool:
    path = Path(run_dir) / name
    return path.is_file() and json.loads(path.read_text()).get("status") == "passed"


def _progress(done: int, total: int, trajectory_id: str) -> None:
    if done == 1 or done == total or done % max(1, total // 20) == 0:
        print(f"{done}/{total} trials: {trajectory_id}", flush=True)


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)


def run(args=None) -> int:
    args = _parser().parse_args(args)
    run_dir = _resolve_run(args)
    if args.command == "serve":
        from .web import serve
        serve(run_dir, host=args.host, port=args.port)
        return 0
    config = _load_config(run_dir)
    print(f"run directory: {run_dir}", flush=True)
    if args.command == "calibrate":
        config, reports = calibrate_run(run_dir, config, args.repeats)
        result = json.loads((run_dir / "calibration" / "summary.json").read_text())
        _json(result)
        return 0 if result.get("status") == "passed" else 2
    if args.command == "preflight":
        config, result = run_preflight(run_dir, config, tuple(args.counts))
        _json(result)
        return 0 if result.get("status") == "passed" else 2
    if args.command == "pilot":
        if not _gate_passed(run_dir, "calibration/summary.json"):
            raise RuntimeError("pilot requires a passing label-free calibration")
        if not _gate_passed(run_dir, "metrics/preflight.json"):
            raise RuntimeError("pilot requires a passing software-only task-design preflight")
        _json(run_pilot(run_dir, config, _progress))
        return 0
    if args.command == "train":
        if not _gate_passed(run_dir, "calibration/summary.json") or not _gate_passed(run_dir, "metrics/preflight.json"):
            raise RuntimeError("final training requires passing calibration and task-design gates")
        _json(run_training_stage(run_dir, config, "final", _progress))
        return 0
    if args.command == "evaluate":
        if not (run_dir / "final" / "metrics" / "training.json").is_file():
            raise RuntimeError("final evaluation requires completed frozen final readouts")
        _json(run_evaluation_stage(run_dir, config, "final", progress=_progress,
                                   test_seed=args.test_seed))
        return 0
    if args.command == "all":
        config, _ = calibrate_run(run_dir, config, args.calibration_repeats)
        calibration = json.loads((run_dir / "calibration" / "summary.json").read_text())
        _json({"calibration": calibration})
        if calibration.get("status") != "passed":
            return 2
        config, preflight = run_preflight(run_dir, config, tuple(args.preflight_counts))
        _json({"preflight": preflight})
        if preflight.get("status") != "passed":
            _json({"status": "stopped_at_design_gate", "run_dir": str(run_dir)})
            return 2
        if not args.skip_pilot:
            _json({"pilot": run_pilot(run_dir, config, _progress)})
        if not args.skip_final:
            _json({"training": run_training_stage(run_dir, config, "final", _progress)})
            _json({"evaluation": run_evaluation_stage(run_dir, config, "final", progress=_progress)})
        return 0
    raise AssertionError(args.command)


def main() -> None:
    try:
        result = run()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
