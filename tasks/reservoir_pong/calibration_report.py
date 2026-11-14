"""Markdown rendering for the label-free calibration report."""

from __future__ import annotations

from pathlib import Path


def write_calibration_report(run_dir: Path, gate: dict,
                             reports: list[ReservoirCalibration],
                             selected: CalibrationCandidate | None,
                             config: ExperimentConfig) -> None:
    lines = [
        "# Label-free reservoir calibration", "",
        f"Status: **{gate['status']}**. {gate.get('reason', '')}".strip(), "",
        "Stimulus settings were screened using calibration measurements only. Task labels were not used.", "",
        "| Subreservoir | Structural hash | Status | Background mean | Background p99 | Max passing gap |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for report in reports:
        maximum = report.usable_maximum_gap_ms
        lines.append(
            f"| {report.slot} | `{report.reservoir_hash}` | {report.status} | "
            f"{report.background_mean:.3f} | {report.background_p99:.3f} | "
            f"{maximum if maximum is not None else '—'} |"
        )
    if selected is not None:
        blank = 2 * selected.phase_ms + config.guard_ms
        lines.extend(["", "## Selected waveform", "",
                      f"{selected.amplitude_ua:g} µA, {selected.phase_ms:g} ms per phase, "
                      f"{selected.window_ms:g} ms feature window, and {blank:g} ms post-onset "
                      "blanking including pulse phases and guard.",
                      f"Usable gap interval: {gate.get('minimum_gap_ms')}–"
                      f"{gate.get('maximum_gap_ms')} ms; fall duration: {gate.get('fall_ms')} ms."])
    for report in reports:
        lines.extend(["", f"## Subreservoir {report.slot}", "",
                      f"Physical 4×4 grid: `{report.electrode_grid}`.", "",
                      "### Spatial calibration", "",
                      "| Phase (ms) | Amplitude (µA) | Window (ms) | Response floor | Location ID | Pass |",
                      "|---:|---:|---:|---:|---:|:---:|"])
        for item in report.candidates:
            lines.append(
                f"| {item.phase_ms:g} | {item.amplitude_ua:g} | {item.window_ms:g} | "
                f"{item.response_rate:.3f} | {item.location_accuracy:.3f} | "
                f"{'yes' if item.passed else 'no'} |"
            )
        if report.memory:
            lines.extend(["", "### Paired-pulse memory", "",
                          "| Phase | Amplitude | Window | Gap (ms) | Four-way accuracy | Bootstrap lower 95% | Pass |",
                          "|---:|---:|---:|---:|---:|---:|:---:|"])
            for item in report.memory:
                lines.append(
                    f"| {item.phase_ms:g} | {item.amplitude_ua:g} | {item.window_ms:g} | "
                    f"{item.gap_ms:g} | {item.accuracy:.3f} | {item.lower_95:.3f} | "
                    f"{'yes' if item.passed else 'no'} |"
                )
        controls = report.controls
        if controls:
            carry = controls.get("previous_ball_label_control", {})
            lines.extend(["", "### Controls", ""])
            if controls.get("no_stimulation") is not None:
                lines.append(f"No-input control: `{controls['no_stimulation']}`.")
            if controls.get("zero_spike_feature_control") is not None:
                lines.append(
                    "Zero-spike feature pipeline passed: "
                    f"{controls['zero_spike_feature_control'].get('passed', False)}."
                )
            if controls.get("status") == "not_run_no_common_configuration":
                lines.append(
                    "Reservoir-dependent controls were not run because no common spatial "
                    "calibration configuration passed."
                )
            artifact = controls.get("synthetic_artifact_blanking_control", {})
            if artifact:
                lines.append(
                    f"Synthetic artifact blanking: {'passed' if artifact.get('passed') else 'failed'}; "
                    f"{artifact.get('artifact_events_injected', 0)} injected events contributed "
                    f"{artifact.get('artifact_feature_sum', 'unknown')} features, and "
                    f"{artifact.get('post_blank_spikes_counted', 0)} valid boundary-control spikes "
                    "were counted. This checks the imposed software rule, not amplifier recovery."
                )
            train = controls.get("repeated_pulse_train", {})
            if train:
                lines.append(
                    f"Repeated {train.get('pulses_per_train', 0)}-pulse train at "
                    f"{train.get('gap_ms', 0):g} ms gaps: first-three mean "
                    f"{train.get('first_three_mean_spikes', 0):.3f}, last-three mean "
                    f"{train.get('last_three_mean_spikes', 0):.3f}, tail mean "
                    f"{train.get('tail_spikes_after_last_counting_window_mean', 0):.3f} spikes."
                )
            if carry:
                lines.append(
                    f"Previous-ball label decoding: {carry.get('accuracy', 0.0):.3f} "
                    f"(chance {carry.get('chance', 0.25):.2f}); maximum matched-seed feature "
                    f"difference {carry.get('maximum_same_seed_feature_difference', 'not measured')}."
                )
            for sensitivity in controls.get("longer_blanking_sensitivity", []):
                lines.append(
                    f"- Longer blanking +{sensitivity['extra_blank_ms']:g} ms: location ID "
                    f"{sensitivity['location_accuracy']:.3f}; this is post hoc sensitivity "
                    "under an imposed simulation blanking rule."
                )
    lines.extend(["", "## Interpretation limit", "",
                  "The blanking guard is an imposed simulation recording rule. Spike-event output does not "
                  "establish biological amplifier recovery. The three deterministic, non-overlapping "
                  "selections are subreservoirs of one simulated culture, not independent biological dishes.", ""])
    (run_dir / "calibration" / "report.md").write_text("\n".join(lines))
