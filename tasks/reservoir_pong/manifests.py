"""Write structural provenance for the fixed simulated reservoirs."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from .records import ExperimentConfig, ExperimentManifest


def write_reservoir_manifest(run_dir: Path, reservoir, config: ExperimentConfig) -> None:
    target = Path(run_dir) / "reservoirs" / f"slot{reservoir.slot}.json"
    manifest = ExperimentManifest(
        config_hash=config.config_hash,
        backend=type(reservoir.env).__module__,
        livn_version=version("livn"),
        reservoir_hash=reservoir.reservoir_hash,
        selected_gids=tuple(int(g) for g in reservoir.selected_gids),
        model_hash=reservoir.model_hash,
        software_version=config.software_version,
        plasticity_enabled=reservoir.plasticity_enabled,
    )
    payload = json.loads(json.dumps(manifest.__dict__))
    payload["selected_gids"] = list(payload["selected_gids"])
    payload["grid"] = reservoir.electrode_grid
    payload["recording_channels"] = [int(channel) for channel in reservoir.env.io.channel_ids]
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
