"""Markdown and JSON artifacts for held-out Pong evaluation."""

from __future__ import annotations

from pathlib import Path

from .evaluation import write_metrics


def write_evaluation_artifacts(stage_dir: Path, stage: str, metrics: dict) -> None:
    stage_dir = Path(stage_dir)
    write_metrics(metrics, stage_dir / "metrics" / "evaluation.json")
    lines = [
        f"# {stage.title()} Pong evaluation", "",
        f"Scientific gate: **{'passed' if metrics['scientific_success'] else 'not passed'}**.",
        "",
        f"Paired opposite-side catch gain: {metrics['mean_catch_gain']:.3f} "
        f"(95% trajectory bootstrap {metrics['gain_ci_lower']:.3f} to "
        f"{metrics['gain_ci_upper']:.3f}); {metrics['opposite_trajectory_count']} "
        "opposite-side trajectories across the matched block.", "",
        "| Subreservoir | Test records | Catch rate | Raw-pulse Ridge | Follow ball | "
        "Opposite-side gain | Neural MAE | Raw MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slot, item in metrics["reservoirs"].items():
        lines.append(
            f"| {slot} | {item['n']} | {item['catch_rate']:.3f} | "
            f"{item['raw_catch_rate']:.3f} | {item['follow_catch_rate']:.3f} | "
            f"{item['opposite_catch_gain']:.3f} | {item['landing_mae']:.3f} | "
            f"{item['raw_landing_mae']:.3f} |"
        )
    remaining_slices = metrics.get("normalized_remaining_error_slices", {})
    if remaining_slices:
        lines.extend([
            "", "## Unclipped landing prediction error by remaining fall distance", "",
            "Each ball contributes one mean per slice after averaging its subreservoir and noise repeats.",
            "| Fall remaining | Balls | Decision windows | Reservoir Ridge MAE | Raw-pulse MAE | Raw − Ridge (95% CI) |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for item in remaining_slices.values():
            if item["neural_mae"] is None:
                continue
            low, high = item["raw_minus_neural_mae_ci_95"]
            lines.append(
                f"| {item['remaining_fraction'][0]:.3f}–{item['remaining_fraction'][1]:.3f} | "
                f"{item['trajectories']} | {item['decisions']} | "
                f"{item['neural_mae']:.3f} | {item['raw_mae']:.3f} | "
                f"{item['raw_minus_neural_mae']:.3f} ({low:.3f}, {high:.3f}) |"
            )
    if "constant_catch_rate" in metrics:
        lines.extend(["", f"Constant training-mean aim: {metrics['constant_catch_rate']:.3f} catch."])
    controls = metrics.get("matched_test_controls", {})
    if controls:
        zero = controls["reservoir_ridge_zero_spike"]
        no_stim = controls["actual_no_stimulation"]
        lines.extend([
            "", "## Matched test controls", "",
            f"All-zero Ridge features: {zero['catch_rate']:.3f} catch, "
            f"MAE {zero['landing_mae']:.3f}; paired catch gain over Ridge "
            f"{zero['catch_gain_vs_reservoir_ridge']:.3f} "
            f"(95% CI {zero['catch_gain_vs_reservoir_ridge_ci_95'][0]:.3f} to "
            f"{zero['catch_gain_vs_reservoir_ridge_ci_95'][1]:.3f}).",
            f"Actual no-stimulation runs: {no_stim['catch_rate']:.3f} catch, "
            f"MAE {no_stim['landing_mae']:.3f}; paired catch gain over Ridge "
            f"{no_stim['catch_gain_vs_reservoir_ridge']:.3f} "
            f"(95% CI {no_stim['catch_gain_vs_reservoir_ridge_ci_95'][0]:.3f} to "
            f"{no_stim['catch_gain_vs_reservoir_ridge_ci_95'][1]:.3f}); mean network "
            f"spikes {no_stim['mean_unstimulated_network_spikes']:.1f} "
            f"(95th percentile {no_stim['p95_unstimulated_network_spikes']:.0f}).",
            "No-stimulation trials delivered no pulses and used the matched trial noise seed.",
        ])
        lines.extend([
            "", "| Subreservoir | Ridge zero-feature catch | MAE | No-stimulation catch | MAE |",
            "|---:|---:|---:|---:|---:|",
        ])
        zero_by_slot = controls.get("zero_spike_subreservoirs", {})
        no_stim_by_slot = controls.get("no_stimulation_subreservoirs", {})
        for slot in sorted(set(zero_by_slot) & set(no_stim_by_slot), key=int):
            zero_slot, no_stim_slot = zero_by_slot[slot], no_stim_by_slot[slot]
            lines.append(
                f"| {slot} | {zero_slot['catch_rate']:.3f} | "
                f"{zero_slot['landing_mae']:.3f} | {no_stim_slot['catch_rate']:.3f} | "
                f"{no_stim_slot['landing_mae']:.3f} |"
            )
    lines.extend(["", "## Gate reasons", ""])
    lines.extend([f"- {reason}" for reason in metrics["success_reasons"]]
                 or ["- All predeclared scientific criteria passed."])
    for slot, control in metrics.get("permutation_controls", {}).items():
        lines.extend(["", f"## Shuffled-label null · subreservoir {slot}", "",
                      f"Observed neural catch: {control['observed_neural_catch_rate']:.3f}; "
                      f"null mean: {control['null_mean_catch_rate']:.3f}; "
                      f"95% null interval: {control['null_95_interval'][0]:.3f}–"
                      f"{control['null_95_interval'][1]:.3f}; "
                      f"empirical one-sided p={control['empirical_one_sided_p']:.4f}."])
    lines.extend(["", "Test records were collected after the training block under the stationary simulator. "
                  "This does not model biological drift. Reservoir gains compare each frozen neural "
                  "readout against its matched raw-pulse Ridge baseline. The 400-trajectory overall "
                  "block excludes any additional opposite-side trajectories used only to reach the "
                  "predeclared 100-per-subreservoir opposite-side analysis.", ""])
    (stage_dir / "metrics" / "evaluation.md").write_text("\n".join(lines))
