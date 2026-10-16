"""Amplitude-screen summaries and the locked selection gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset_manifest import RecordingManifestRow
from .fixed_reservoir import ScreenSelection, TrialSpec, build_trial_specs, select_screen_groups


class ScreeningGateFailed(RuntimeError):
    """No screened amplitude met the pre-registered development gate."""


@dataclass(frozen=True)
class AmplitudeScreenSummary:
    amplitude_ua: float
    baseline_firing_rate: float
    during_firing_rate: float
    post_firing_rate: float
    control_during_firing_rate: float
    active_electrode_fraction: float
    positive_control_balanced_accuracy: float
    shuffled_null_balanced_accuracy: float
    early_balanced_accuracy: tuple[float, float]
    early_standard_error: float
    saturated: bool = False

    @property
    def response_is_distinguishable(self) -> bool:
        # A conservative, data-driven screen: the during response must exceed
        # the no-stimulus control by at least 2 control-rate standard-error
        # units; the caller can encode the uncertainty in the rate itself.
        return self.during_firing_rate > self.control_during_firing_rate

    @property
    def positive_control_passes(self) -> bool:
        return self.positive_control_balanced_accuracy > self.shuffled_null_balanced_accuracy

    @property
    def early_score(self) -> float:
        return float(np.mean(self.early_balanced_accuracy))


def choose_amplitude(summaries: Sequence[AmplitudeScreenSummary]) -> AmplitudeScreenSummary:
    candidates = [summary for summary in summaries if summary.response_is_distinguishable and summary.positive_control_passes and not summary.saturated]
    if not candidates:
        raise ScreeningGateFailed("no amplitude showed promising post-stimulus separation; do not fall back to 100 uA")
    candidates.sort(key=lambda summary: (summary.early_score, -summary.amplitude_ua), reverse=True)
    best = candidates[0]
    tied = [summary for summary in candidates if abs(summary.early_score - best.early_score) <= max(summary.early_standard_error, best.early_standard_error)]
    return min(tied, key=lambda summary: summary.amplitude_ua)


def build_screen_schedule(rows: Sequence[RecordingManifestRow], reservoir_hash: str, amplitudes: Sequence[float] = (1.5, 3.125, 6.25, 12.5, 25.0), second_pass_amplitudes: Sequence[float] = ()) -> tuple[list[TrialSpec], ScreenSelection]:
    """Create the preregistered 240 + 96 + 48 screening/control schedule."""
    selection = select_screen_groups(rows, 24)
    by_id = {row.recording_id: row for row in rows}
    selected_rows = [by_id[recording_id] for recording_id in selection.recording_ids]
    first_pass: list[TrialSpec] = []
    for amplitude in amplitudes:
        first_pass.extend(build_trial_specs(selected_rows, reservoir_hash, amplitude, 1))
    second_pass: list[TrialSpec] = []
    for amplitude in second_pass_amplitudes:
        second_pass.extend(build_trial_specs(selected_rows, reservoir_hash, amplitude, 2))
        # Keep only the second stochastic realization in pass 2.
        second_pass = [spec for spec in second_pass if spec.repeat == 1]
    controls = [spec for spec in build_trial_specs(selected_rows, reservoir_hash, 0.0, 1, include_controls=True) if spec.is_control]
    schedule = first_pass + second_pass + controls
    if len(schedule) > 384:
        raise ValueError("screening schedule exceeds the preregistered 384-trial maximum")
    return schedule, selection
