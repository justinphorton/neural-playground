"""Three-stage fixed-reservoir C4-vs-E4 experiment.

This module deliberately keeps simulation, feature extraction, and analysis as
separate interfaces.  It does not install a model into the existing website.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import multiprocessing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import joblib
from livn.env import Env

from .analysis import grouped_folds, evaluate_features, make_classifier
from .audio_processing import AudioAudit, audit_audio, prepare_audio
from .audio_diagnostics import audio_rms_diagnostic
from .dataset_manifest import RecordingManifestRow, build_training_manifest, write_manifest
from .experiment_config import ExperimentConfig, create_experiment_directory
from .feature_extraction import FeatureDataset, PRIMARY_MEMORY_WINDOWS_MS, extract_features
from .raw_trials import RawTrial, load_raw_trial, save_raw_trial
from .reservoir_artifact import (
    ReservoirManifest,
    assert_reservoir_unchanged,
    create_canonical_reservoir,
    load_canonical_reservoir,
    runtime_reservoir_hash,
)
from .stimulus import build_trial_current


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    recording_id: str
    wav_path: str | None
    label: str
    condition_group_id: str | None
    amplitude_ua: float
    repeat: int
    noise_seed: int
    expected_reservoir_hash: str
    is_control: bool = False


@dataclass(frozen=True)
class ScreenSelection:
    recording_ids: tuple[str, ...]
    condition_groups: tuple[str, ...]


def deterministic_seed(recording_id: str, repeat: int, base_seed: int = 10_000, amplitude_ua: float = 0.0) -> int:
    digest = hashlib.sha256(f"{recording_id}:{repeat}:{base_seed}:{amplitude_ua:g}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def select_screen_groups(rows: Sequence[RecordingManifestRow], n_groups: int = 24) -> ScreenSelection:
    groups: dict[str, list[RecordingManifestRow]] = {}
    for row in rows:
        groups.setdefault(row.condition_group_id, []).append(row)
    eligible = []
    for group, members in groups.items():
        if {member.note for member in members} == {"C4", "E4"}:
            eligible.append((group, members))
    if len(eligible) < n_groups:
        raise ValueError(f"need {n_groups} paired groups, found {len(eligible)}")
    # Round-robin across family/hold buckets makes the screen balanced and deterministic.
    buckets: dict[tuple[str, int], list[tuple[str, list[RecordingManifestRow]]]] = {}
    for item in sorted(eligible, key=lambda item: item[0]):
        member = item[1][0]
        buckets.setdefault((member.piano_family, member.hold_duration_ms), []).append(item)
    selected: list[tuple[str, list[RecordingManifestRow]]] = []
    bucket_keys = sorted(buckets)
    while len(selected) < n_groups:
        progressed = False
        for key in bucket_keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
                if len(selected) == n_groups:
                    break
        if not progressed:
            break
    if len(selected) != n_groups:
        raise ValueError("not enough balanced screen groups")
    ids = tuple(row.recording_id for _, members in selected for row in sorted(members, key=lambda row: row.note))
    return ScreenSelection(recording_ids=ids, condition_groups=tuple(group for group, _ in selected))


def build_trial_specs(rows: Sequence[RecordingManifestRow], reservoir_hash: str, amplitude_ua: float, repeats: int, include_controls: bool = False, base_seed: int = 10_000) -> list[TrialSpec]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    specs: list[TrialSpec] = []
    for row in rows:
        for repeat in range(repeats):
            specs.append(TrialSpec(
                trial_id=f"{row.recording_id}_r{repeat:02d}_{amplitude_ua:g}uA",
                recording_id=row.recording_id, wav_path=row.path, label=row.note,
                condition_group_id=row.condition_group_id, amplitude_ua=float(amplitude_ua), repeat=repeat,
                noise_seed=deterministic_seed(row.recording_id, repeat, base_seed, amplitude_ua),
                expected_reservoir_hash=reservoir_hash,
            ))
    if include_controls:
        for index, row in enumerate(rows):
            specs.append(TrialSpec(
                trial_id=f"control_{index:04d}_{amplitude_ua:g}uA", recording_id=f"control_{index:04d}",
                wav_path=None, label="CONTROL", condition_group_id=row.condition_group_id,
                amplitude_ua=0.0, repeat=0, noise_seed=deterministic_seed(f"control_{index:04d}", 0, base_seed, 0.0),
                expected_reservoir_hash=reservoir_hash, is_control=True,
            ))
    seeds = [spec.noise_seed for spec in specs]
    if len(seeds) != len(set(seeds)):
        raise AssertionError("independent trials must have unique noise seeds")
    return specs


def trial_checkpoint_matches(path: Path, spec: TrialSpec, config_hash: str) -> bool:
    """A raw checkpoint is reusable only when every identity field agrees."""
    try:
        trial = load_raw_trial(path)
    except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError):
        return False
    return (
        trial.trial_id == spec.trial_id
        and trial.recording_id == spec.recording_id
        and trial.label == spec.label
        and trial.repeat == spec.repeat
        and trial.noise_seed == spec.noise_seed
        and trial.reservoir_hash == spec.expected_reservoir_hash
        and trial.metadata.get("config_hash") == config_hash
        and float(trial.metadata.get("amplitude_ua", spec.amplitude_ua)) == float(spec.amplitude_ua)
    )


def atomic_save_trial(trial: RawTrial, path: Path) -> None:
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    save_raw_trial(trial, temporary)
    temporary.replace(path)


def assert_single_reservoir_hash(trials: Sequence[RawTrial], expected_hash: str | None = None) -> str:
    hashes = {trial.reservoir_hash for trial in trials}
    if len(hashes) != 1:
        raise AssertionError(f"dataset contains multiple reservoir hashes: {sorted(hashes)}")
    value = next(iter(hashes))
    if expected_hash is not None and value != expected_hash:
        raise AssertionError(f"dataset reservoir hash {value} does not match expected {expected_hash}")
    return value


def _preflight_worker(artifact_dir: str) -> str:
    env, manifest, site = load_canonical_reservoir(Path(artifact_dir))
    try:
        return runtime_reservoir_hash(env, site, np.asarray(manifest.selected_gids, dtype=np.int64))
    finally:
        env.close()


def preflight_worker_hashes(artifact_dir: Path, worker_count: int = 8) -> tuple[str, ...]:
    """Launch real worker processes that load (but do not simulate) the artifact."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    context = multiprocessing.get_context("spawn")
    with context.Pool(worker_count) as pool:
        hashes = tuple(pool.map(_preflight_worker, [str(artifact_dir)] * worker_count))
    if len(set(hashes)) != 1:
        raise RuntimeError(f"worker reservoir hashes differ: {hashes}")
    manifest = ReservoirManifest.read(Path(artifact_dir) / "reservoir_manifest.json")
    if hashes[0] != manifest.reservoir_hash:
        raise RuntimeError("worker hash differs from configured reservoir hash")
    return hashes


def _collect_worker(args) -> tuple[int, int, int]:
    worker_id, specs, config_payload, experiment_dir, reservoir_dir = args
    config = ExperimentConfig.from_dict(config_payload)
    stage = SimulationStage(config, Path(experiment_dir), Path(reservoir_dir))
    collected = resumed = 0
    try:
        for spec in specs:
            path = Path(experiment_dir) / "raw_trials" / f"{spec.trial_id}.npz"
            if trial_checkpoint_matches(path, spec, config.config_hash):
                resumed += 1
                continue
            trial = stage.run_trial(spec)
            atomic_save_trial(trial, path)
            collected += 1
    finally:
        stage.close()
    return worker_id, collected, resumed


def collect_trials_parallel(specs: Sequence[TrialSpec], config: ExperimentConfig, experiment_dir: Path, reservoir_dir: Path, worker_count: int | None = None) -> list[tuple[int, int, int]]:
    """Collect a resumable set of raw trials with isolated canonical-reservoir workers."""
    if not specs:
        return []
    if len({spec.noise_seed for spec in specs}) != len(specs):
        raise ValueError("trial schedule contains duplicate noise seeds")
    worker_count = worker_count or config.workers
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    shards = [list(chunk) for chunk in np.array_split(np.asarray(specs, dtype=object), min(worker_count, len(specs)))]
    args = [(index, shard, config.to_dict(), str(experiment_dir), str(reservoir_dir)) for index, shard in enumerate(shards)]
    context = multiprocessing.get_context("spawn")
    with context.Pool(len(args)) as pool:
        results = pool.map(_collect_worker, args)
    return sorted(results)


def _write_audio_audits(rows: Sequence[RecordingManifestRow], path: Path, peak_current_ua: float, config: ExperimentConfig) -> list[AudioAudit]:
    audits = [audit_audio(row.path, peak_current_ua, config.sample_rate_hz, config.fade_out_ms) for row in rows]
    with Path(path).open("w", newline="") as handle:
        if audits:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(audits[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(audit) for audit in audits)
    return audits


class SimulationStage:
    def __init__(self, config: ExperimentConfig, experiment_dir: Path, reservoir_dir: Path):
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.reservoir_dir = Path(reservoir_dir)
        self._env = None
        self._manifest = None
        self._site = None

    def _reservoir(self):
        if self._env is None:
            self._env, self._manifest, self._site = load_canonical_reservoir(self.reservoir_dir)
        return self._env, self._manifest, self._site

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._manifest = None
            self._site = None

    def run_trial(self, spec: TrialSpec, output_path: Path | None = None, save_waveform: bool = False) -> RawTrial:
        env, manifest, site = self._reservoir()
        try:
            before = runtime_reservoir_hash(env, site, np.asarray(manifest.selected_gids, dtype=np.int64))
            if before != spec.expected_reservoir_hash:
                raise RuntimeError("trial spec references a different reservoir")
            env.clear(reseed=False)
            env.disable_plasticity()
            env.reseed_noise(spec.noise_seed)
            if spec.is_control:
                waveform = np.zeros(round(self.config.sample_rate_hz * self.config.stimulation_ms / 1000), dtype=np.float32)
                audio_audit = None
            else:
                audio_audit = asdict(audit_audio(Path(spec.wav_path), spec.amplitude_ua, self.config.sample_rate_hz, self.config.fade_out_ms))
                waveform = prepare_audio(Path(spec.wav_path), self.config.sample_rate_hz).samples
            current, current_diag = build_trial_current(
                waveform, spec.amplitude_ua if not spec.is_control else 0.0,
                baseline_ms=self.config.baseline_ms, post_stimulus_ms=self.config.post_stimulus_ms,
                sample_rate_hz=self.config.sample_rate_hz, fade_out_ms=self.config.fade_out_ms,
                stimulation_electrode_index=site.stimulus_index,
                electrode_count=env.io.num_channels,
            )
            coordinates = env.system.transform_coordinates(env.model.stimulus_coordinates, populations=env.active_populations())
            coordinates = coordinates[np.isin(coordinates[:, 0].astype(np.int64), env.simulated_gids())]
            stimulus = env.io.cell_stimulus(coordinates, current, dt=0.1)
            env.record_spikes()
            response = env.run(self.config.baseline_ms + self.config.stimulation_ms + self.config.post_stimulus_ms, stimulus=stimulus)
            spike_ids = np.asarray(response.spike_ids, dtype=np.int64)
            spike_times = np.asarray(response.spike_times, dtype=float)
            _, by_channel = env.channel_recording(response.spike_ids, response.spike_times)
            event_channels: list[int] = []
            event_times: list[float] = []
            for channel in site.neighbor_ids:
                values = np.asarray(by_channel.get(channel, []), dtype=float)
                event_channels.extend([channel] * len(values))
                event_times.extend(values.tolist())
            order = np.argsort(event_times, kind="stable")
            after = runtime_reservoir_hash(env, site, np.asarray(manifest.selected_gids, dtype=np.int64))
            assert_reservoir_unchanged(env, site, spec.expected_reservoir_hash, np.asarray(manifest.selected_gids, dtype=np.int64))
            metadata = {
                "experiment_id": self.experiment_dir.name, "wav_path": spec.wav_path,
                "condition_group_id": spec.condition_group_id,
                "audio_audit": audio_audit,
                "amplitude_ua": spec.amplitude_ua, "configured_current": current_diag,
                "electrodes": {"stimulation": site.stimulus_id, "recording": list(site.neighbor_ids)},
                "fade_out_ms": self.config.fade_out_ms, "config_hash": self.config.config_hash,
                "software_version": self.config.software_version, "livn_version": importlib.metadata.version("livn"),
                "backend": manifest.backend,
                "phase_timings_ms": [self.config.baseline_ms, self.config.stimulation_ms, self.config.post_stimulus_ms],
                "pre_trial_reservoir_hash": before, "post_trial_reservoir_hash": after,
                "raw_event_count": len(event_times), "stimulus_hash": current_diag["stimulus_sha256"],
            }
            trial = RawTrial(
                trial_id=spec.trial_id, recording_id=spec.recording_id, label=spec.label, repeat=spec.repeat,
                reservoir_hash=spec.expected_reservoir_hash, noise_seed=spec.noise_seed,
                neuron_spike_times_ms=spike_times, neuron_spike_ids=spike_ids,
                event_times_ms=np.asarray(event_times, dtype=float)[order], event_channel_ids=np.asarray(event_channels, dtype=np.int32)[order],
                phase_timings_ms=(float(self.config.baseline_ms), float(self.config.stimulation_ms), float(self.config.post_stimulus_ms)), metadata=metadata,
            )
            if save_waveform:
                waveform_path = self.experiment_dir / "raw_trials" / f"{spec.trial_id}_stimulus.npy"
                waveform_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(waveform_path, current)
            if output_path is not None:
                save_raw_trial(trial, output_path)
            return trial
        finally:
            # The worker owns the cached environment and closes it after its shard.
            pass


class FeatureStage:
    def __init__(self, recording_electrodes: Sequence[int]):
        self.recording_electrodes = tuple(int(value) for value in recording_electrodes)

    def from_trials(self, trials: Sequence[RawTrial], window_ms: tuple[float, float], mode: str = "fine_temporal") -> FeatureDataset:
        if any(window_ms[0] < 0 or window_ms[1] > 1000 for _ in [0]):
            raise ValueError("feature windows must be inside the post-stimulus second")
        rows = [extract_features(
            trial.event_channel_ids,
            trial.event_times_ms - float(trial.phase_timings_ms[0] + trial.phase_timings_ms[1]),
            self.recording_electrodes, window_ms, mode,
        ) for trial in trials]
        return FeatureDataset(
            X=np.stack(rows), labels=np.asarray([trial.label for trial in trials]),
            recording_ids=np.asarray([trial.recording_id for trial in trials]),
            condition_groups=np.asarray([trial.metadata.get("condition_group_id", trial.recording_id) for trial in trials]),
            feature_mode=mode, window_ms=tuple(window_ms), recording_electrodes=self.recording_electrodes,
            is_control=np.asarray([trial.label == "CONTROL" for trial in trials], dtype=bool),
        )


class AnalysisStage:
    def evaluate(self, dataset: FeatureDataset, n_splits: int = 5):
        keep = np.ones(len(dataset.labels), dtype=bool) if dataset.is_control is None else ~dataset.is_control
        labels, recording_ids, groups = dataset.labels[keep], dataset.recording_ids[keep], dataset.condition_groups[keep]
        folds = grouped_folds(labels, recording_ids, groups, n_splits=n_splits)
        return evaluate_features(dataset.X[keep], labels, folds)

    def fit_all(self, dataset: FeatureDataset):
        model = make_classifier()
        keep = np.ones(len(dataset.labels), dtype=bool) if dataset.is_control is None else ~dataset.is_control
        model.fit(dataset.X[keep], dataset.labels[keep])
        return model

    def save_model(self, model, dataset: FeatureDataset, path: Path, *, config: ExperimentConfig, reservoir_hash: str, amplitude_ua: float) -> None:
        """Persist the all-data development model with its immutable provenance."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        metadata = {
            "status": "development",
            "feature_mode": dataset.feature_mode,
            "window_ms": list(dataset.window_ms),
            "recording_electrodes": list(dataset.recording_electrodes),
            "reservoir_hash": reservoir_hash,
            "config_hash": config.config_hash,
            "selected_amplitude_ua": amplitude_ua,
            "class_order": list(getattr(model[-1], "classes_", ("C4", "E4"))),
            "preprocessing": "StandardScaler -> LogisticRegression(C=1, class_weight=balanced, max_iter=2000, random_state=0)",
        }
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def initialize_experiment(root: Path, training_data: Path, config: ExperimentConfig | None = None, name: str = "fixed-c4-e4-memory") -> tuple[Path, list[RecordingManifestRow], ReservoirManifest]:
    config = config or ExperimentConfig()
    rows = build_training_manifest(training_data)
    experiment_dir = create_experiment_directory(root, name, config)
    write_manifest(rows, experiment_dir / "dataset_manifest.csv")
    audits = _write_audio_audits(rows, experiment_dir / "audio_audit.csv", config.default_peak_current_ua, config)
    (experiment_dir / "metrics" / "audio_rms_diagnostic.json").write_text(json.dumps(audio_rms_diagnostic(rows, audits), indent=2, sort_keys=True) + "\n")
    reservoir = create_canonical_reservoir(experiment_dir / "reservoir", seed=config.structural_reservoir_seed, n_cells=config.reservoir_cells)
    return experiment_dir, rows, reservoir
