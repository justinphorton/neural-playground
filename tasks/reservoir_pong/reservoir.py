"""Fixed, explicitly native livn reservoirs and continuous sequence runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LIVN_BACKEND", "native")
if os.environ.get("LIVN_BACKEND") != "native":
    raise RuntimeError("Reservoir Pong requires LIVN_BACKEND=native")

import numpy as np
from livn.backend.config import backend
from livn.env import Env
from livn.system import System, predefined

from .records import ExperimentConfig, structural_hash
from .stimulus import PulseEvent, make_waveform, select_grid


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Reservoir:
    env: Env
    slot: int
    selected_gids: np.ndarray
    electrode_grid: list[tuple[int, float, float]]
    reservoir_hash: str
    model_hash: str
    plasticity_enabled: bool = False

    def close(self) -> None:
        self.env.close()


@dataclass(frozen=True)
class SimulationResult:
    spike_gids: np.ndarray
    spike_times_ms: np.ndarray
    spike_channels: np.ndarray
    spike_channel_times_ms: np.ndarray
    duration_ms: float
    noise_seed: int


def _coordinate_table(system) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    populations = []
    for population in system.populations:
        coordinates = np.asarray(system.coordinate_array(population))
        if coordinates.ndim != 2 or coordinates.shape[1] < 3:
            raise RuntimeError(f"livn returned invalid coordinates for {population}")
        populations.append((population, coordinates))
    if not populations:
        raise RuntimeError("the selected livn system has no cell coordinates")
    return np.vstack([coords for _, coords in populations]), populations


def select_electrode_grid_for_cells(
    electrode_coordinates: np.ndarray,
    cell_coordinates: np.ndarray,
    selected_gids: np.ndarray,
    side: int = 4,
) -> list[tuple[int, float, float]]:
    """Center a physical MEA grid on the selected cells, not the whole culture."""
    cells = np.asarray(cell_coordinates)
    gids = np.asarray(selected_gids, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] < 3 or gids.ndim != 1 or not len(gids):
        raise ValueError("selected cells require an ID vector and an Nx3+ coordinate table")
    selected = np.isin(cells[:, 0].astype(np.int64), gids)
    if int(selected.sum()) != len(gids):
        raise ValueError("coordinate table does not contain every selected neuron exactly once")
    center_xy = np.asarray(cells[selected, 1:3], dtype=float).mean(axis=0)
    return select_grid(electrode_coordinates, tuple(center_xy.tolist()), side)


def select_reservoir_gids(
    system, n_cells: int = 600, count: int = 3, seed: int = 123
) -> list[np.ndarray]:
    """Choose a central livn patch and two disjoint spatial subreservoirs."""
    if n_cells <= 0 or count <= 0:
        raise ValueError("n_cells and count must be positive")
    if isinstance(system, (str, Path)):
        system = System(str(system))
    central = system.selection(n_cells, method="patch", seed=seed)
    used = {int(gid) for gids in central.values() for gid in gids}
    selections = [np.asarray(sorted(used), dtype=np.int64)]
    all_coords, by_population = _coordinate_table(system)
    gids = all_coords[:, 0].astype(np.int64)
    centers = (np.quantile(all_coords[:, 2], np.linspace(0.2, 0.8, count - 1))
               if count > 1 else np.zeros(0))
    x_mid = float((all_coords[:, 1].min() + all_coords[:, 1].max()) / 2.0)
    extent = max(float(np.ptp(all_coords[:, 1])), float(np.ptp(all_coords[:, 2])))
    population_ranges = system.cells_meta_data.population_ranges

    for slot in range(1, count):
        center_y = float(centers[slot - 1])
        eligible = ~np.isin(gids, np.fromiter(used, dtype=np.int64))
        candidates = all_coords[eligible]
        if len(candidates) < n_cells:
            raise RuntimeError("not enough disjoint neurons to build requested subreservoirs")
        distance = ((candidates[:, 1] - x_mid) / max(extent, 1.0)) ** 2
        distance += ((candidates[:, 2] - center_y) / max(extent, 1.0)) ** 2
        chosen = candidates[np.lexsort((candidates[:, 0], distance))[:n_cells], 0].astype(np.int64)
        by_population_selection = {}
        for population, start_count in population_ranges.items():
            start, n = start_count
            local = chosen[(chosen >= start) & (chosen < start + n)]
            if len(local):
                by_population_selection[population] = np.sort(local)
        selection = np.concatenate(list(by_population_selection.values())) if by_population_selection else np.array([], dtype=np.int64)
        if len(selection) != n_cells:
            raise RuntimeError(f"reservoir {slot} selection has {len(selection)} cells, expected {n_cells}")
        selections.append(np.sort(selection))
        used.update(int(gid) for gid in selection)

    for i, left in enumerate(selections):
        for right in selections[i + 1:]:
            if np.intersect1d(left, right).size:
                raise RuntimeError("reservoir selections overlap")
    return selections


def _selection_by_population(system, gids: np.ndarray) -> dict[str, list[int]]:
    selected = {}
    ranges = system.cells_meta_data.population_ranges
    for population, (start, count) in ranges.items():
        local = gids[(gids >= start) & (gids < start + count)]
        if len(local):
            selected[population] = local.astype(int).tolist()
    return selected


def build_reservoir(
    config: ExperimentConfig,
    slot: int = 0,
    gids: np.ndarray | None = None,
    cache_dir: Path | None = None,
) -> Reservoir:
    if backend() != "native":
        raise RuntimeError(f"livn selected {backend()!r}; expected the native simulator")
    cache_dir = Path(cache_dir or REPO_ROOT)
    system = System(predefined("EI", download_directory=str(cache_dir)))
    selections = select_reservoir_gids(system, config.n_cells, config.reservoir_count, config.seed)
    if not 0 <= slot < len(selections):
        raise ValueError(f"slot {slot} is outside available reservoirs 0..{len(selections)-1}")
    selected = np.asarray(selections[slot] if gids is None else gids, dtype=np.int64)
    if len(selected) != config.n_cells or len(np.unique(selected)) != config.n_cells:
        raise ValueError(f"reservoir must have exactly {config.n_cells} unique cells")
    env = Env(system, seed=config.seed + slot)
    try:
        all_coordinates, _ = _coordinate_table(system)
        env.selection(_selection_by_population(system, selected), method="first")
        env.init().apply_model_defaults()
        env.disable_plasticity()
        env.record_spikes()
        if getattr(env, "_plasticity_enabled", False):
            raise RuntimeError("livn plasticity must be disabled for this experiment")
        grid = select_electrode_grid_for_cells(
            np.asarray(env.io.electrode_coordinates), all_coordinates, selected, side=4,
        )
        structural = structural_hash(env, selected, grid)
        model_hash = structural
        return Reservoir(env, slot, np.sort(selected), grid, structural, model_hash, False)
    except BaseException:
        env.close()
        raise


class ReservoirRunner:
    def __init__(self, reservoir: Reservoir, config: ExperimentConfig):
        self.reservoir = reservoir
        self.config = config
        self.channel_ids = tuple(int(ch) for ch in reservoir.env.io.channel_ids)

    def simulate(
        self,
        events: list[PulseEvent],
        duration_ms: float,
        noise_seed: int,
        chunk_ms: float | None = None,
    ) -> SimulationResult:
        config, env = self.config, self.reservoir.env
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        dt = config.stimulus_dt_ms
        n_steps = round(duration_ms / dt)
        if not np.isclose(n_steps * dt, duration_ms, atol=1e-8):
            raise ValueError("duration must be an integral number of stimulus samples")
        waveform = make_waveform(events, duration_ms, len(self.channel_ids), dt, self.channel_ids)
        chunk_steps = n_steps if chunk_ms is None else round(chunk_ms / dt)
        if chunk_steps <= 0 or not np.isclose(chunk_steps * dt, chunk_ms or duration_ms, atol=1e-8):
            raise ValueError("chunk_ms must be an integral number of positive stimulus samples")
        env.clear(reseed=False)
        env.reseed_noise(int(noise_seed))
        ids: list[np.ndarray] = []
        times: list[np.ndarray] = []
        offset_steps = 0
        while offset_steps < n_steps:
            end_steps = min(n_steps, offset_steps + chunk_steps)
            current = env.cell_stimulus(waveform[offset_steps:end_steps], dt=dt)
            out = env.run((end_steps - offset_steps) * dt, current)
            gids = np.asarray(out.spike_ids, dtype=np.int64)
            local_times = np.asarray(out.spike_times, dtype=float) + offset_steps * dt
            ids.append(gids)
            times.append(local_times)
            offset_steps = end_steps
        all_ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        all_times = np.concatenate(times) if times else np.zeros(0, dtype=float)
        # Recompute channel IDs once over the complete recording so chunk
        # boundaries and livn's returned channel ordering cannot misalign spikes.
        _, channel_times = env.channel_recording(all_ids, all_times)
        ch_parts, time_parts = [], []
        for channel in self.channel_ids:
            values = np.asarray(channel_times.get(channel, np.zeros(0)), dtype=float)
            ch_parts.append(np.full(len(values), channel, dtype=np.int32))
            time_parts.append(values)
        event_channels = np.concatenate(ch_parts) if ch_parts else np.zeros(0, dtype=np.int32)
        event_times = np.concatenate(time_parts) if time_parts else np.zeros(0, dtype=float)
        order = np.argsort(event_times, kind="stable")
        return SimulationResult(all_ids, all_times, event_channels[order], event_times[order], duration_ms, int(noise_seed))
