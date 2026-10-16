"""Canonical saved EI reservoir and identity checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from livn.env import Env
from livn.system import predefined

from .reservoir_pilot import N_CELLS, REPO_ROOT, ElectrodeSite, choose_site


def _hash_bytes(digest: hashlib._Hash, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _source_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(Path(root).glob("systems/graphs/EI/*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result[str(path.relative_to(root))] = digest
    return result


def runtime_reservoir_hash(env, site: ElectrodeSite, selected_gids: np.ndarray | None = None) -> str:
    digest = hashlib.sha256()
    _hash_bytes(digest, np.asarray(env.simulated_gids() if selected_gids is None else selected_gids, dtype=np.int64))
    _hash_bytes(digest, np.asarray([site.stimulus_index, site.stimulus_id, *site.neighbor_ids], dtype=np.int64))
    if hasattr(env.io, "electrode_coordinates"):
        _hash_bytes(digest, np.asarray(env.io.electrode_coordinates))
    if hasattr(env, "simulated_coordinates"):
        _hash_bytes(digest, np.asarray(env.simulated_coordinates()))
    if env.conn is not None:
        for name in ("pre_gid", "syn_row", "post_pop", "pre_pop", "dest_sectype", "mech_id", "delay", "weight", "wslot"):
            if hasattr(env.conn, name):
                _hash_bytes(digest, getattr(env.conn, name))
    # ``Env._w`` is a solver scratch/state buffer and changes during an
    # otherwise non-plastic simulation.  The connection table's weights are
    # the structural values that must remain fixed, so only those enter this
    # identity hash.
    digest.update(type(env.model).__module__.encode())
    digest.update(type(env.model).__qualname__.encode())
    digest.update(importlib.metadata.version("livn").encode())
    backend = getattr(getattr(env, "_lib", None), "__class__", type(None)).__module__
    digest.update(str(backend).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReservoirManifest:
    selected_gids: tuple[int, ...]
    selected_gids_by_population: dict[str, tuple[int, ...]]
    source_file_hashes: dict[str, str]
    livn_version: str
    backend: str
    fixed_simulation_seed: int
    model_parameters: dict[str, Any]
    electrode_geometry: list[list[float]]
    stimulation_electrode: int
    recording_electrodes: tuple[int, ...]
    reservoir_hash: str
    n_cells: int = N_CELLS
    plasticity_enabled: bool = False

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> "ReservoirManifest":
        payload = json.loads(Path(path).read_text())
        payload["selected_gids"] = tuple(payload["selected_gids"])
        payload["selected_gids_by_population"] = {key: tuple(value) for key, value in payload["selected_gids_by_population"].items()}
        payload["recording_electrodes"] = tuple(payload["recording_electrodes"])
        return cls(**payload)


def _selection_by_population(env) -> dict[str, tuple[int, ...]]:
    selection = getattr(env, "_selection", None)
    if selection is None:
        raise RuntimeError("canonical artifact requires an explicit cell selection")
    return {str(population): tuple(int(gid) for gid in gids) for population, gids in selection.items()}


def create_canonical_reservoir(artifact_dir: Path, seed: int = 123, n_cells: int = N_CELLS) -> ReservoirManifest:
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"reservoir artifact directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_root = artifact_dir / "source"
    source_root.mkdir()
    source_graph = REPO_ROOT / "systems" / "graphs" / "EI"
    (source_root / "systems" / "graphs").mkdir(parents=True)
    shutil.copytree(source_graph, source_root / "systems" / "graphs" / "EI")
    env = Env(predefined("EI", download_directory=str(source_root)), seed=seed)
    try:
        env.selection(n_cells, method="patch")
        env.init().apply_model_defaults()
        env.disable_plasticity()
        site = choose_site(env.io.electrode_coordinates, env.channel_reach())
        selected_by_pop = _selection_by_population(env)
        selected = tuple(sorted(gid for gids in selected_by_pop.values() for gid in gids))
        if len(selected) != n_cells:
            raise RuntimeError(f"canonical selection returned {len(selected)} cells; expected {n_cells}")
        before = runtime_reservoir_hash(env, site, np.asarray(selected, dtype=np.int64))
        model_parameters = {
            "model": type(env.model).__qualname__,
            "temperature_celsius": getattr(env, "celsius", None),
            "initial_voltage": getattr(env, "v_init", None),
        }
        manifest = ReservoirManifest(
            selected_gids=selected,
            selected_gids_by_population=selected_by_pop,
            source_file_hashes=_source_hashes(source_root),
            livn_version=importlib.metadata.version("livn"),
            backend=str(getattr(getattr(env, "_lib", None), "__class__", type(None)).__module__),
            fixed_simulation_seed=seed, model_parameters=model_parameters,
            electrode_geometry=np.asarray(env.io.electrode_coordinates).tolist(),
            stimulation_electrode=site.stimulus_id,
            recording_electrodes=tuple(site.neighbor_ids), reservoir_hash=before,
            n_cells=n_cells, plasticity_enabled=False,
        )
        manifest.write(artifact_dir / "reservoir_manifest.json")
        (artifact_dir / "selection.json").write_text(json.dumps({k: list(v) for k, v in selected_by_pop.items()}, indent=2) + "\n")
        return manifest
    finally:
        env.close()


def load_canonical_reservoir(artifact_dir: Path) -> tuple[Env, ReservoirManifest, ElectrodeSite]:
    artifact_dir = Path(artifact_dir)
    manifest = ReservoirManifest.read(artifact_dir / "reservoir_manifest.json")
    source_root = artifact_dir / "source"
    env = Env(predefined("EI", download_directory=str(source_root)), seed=manifest.fixed_simulation_seed)
    try:
        env.selection({key: list(value) for key, value in manifest.selected_gids_by_population.items()}, method="first")
        env.init().apply_model_defaults()
        env.disable_plasticity()
        site = choose_site(env.io.electrode_coordinates, env.channel_reach())
        actual = runtime_reservoir_hash(env, site, np.asarray(manifest.selected_gids, dtype=np.int64))
        if actual != manifest.reservoir_hash:
            env.close()
            raise RuntimeError(f"canonical reservoir hash mismatch: expected {manifest.reservoir_hash}, got {actual}")
        if getattr(env, "_plasticity_enabled", False):
            env.close()
            raise RuntimeError("plasticity must be disabled")
        return env, manifest, site
    except BaseException:
        if getattr(env, "_sim", None) is not None:
            env.close()
        raise


def assert_reservoir_unchanged(env, site: ElectrodeSite, expected_hash: str, selected_gids: np.ndarray | None = None) -> None:
    if getattr(env, "_plasticity_enabled", False):
        raise AssertionError("plasticity was enabled during a fixed-reservoir trial")
    actual = runtime_reservoir_hash(env, site, selected_gids)
    if actual != expected_hash:
        raise AssertionError(f"reservoir changed during trial: {expected_hash} -> {actual}")
