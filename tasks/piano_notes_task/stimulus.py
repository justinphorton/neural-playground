"""Exact 10 kHz stimulus construction and scalar diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np


def apply_fade_out(waveform: np.ndarray, sample_rate_hz: int = 10_000, fade_out_ms: float = 15.0) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float64).copy()
    n = int(round(sample_rate_hz * fade_out_ms / 1000.0))
    if n < 1 or n > len(waveform):
        raise ValueError("fade duration does not fit waveform")
    phase = np.arange(1, n + 1, dtype=np.float64) / n
    waveform[-n:] *= 0.5 * (1.0 + np.cos(np.pi * phase))
    waveform[-1] = 0.0
    return waveform.astype(np.float32)


def build_trial_current(
    waveform: np.ndarray,
    peak_current_ua: float,
    baseline_ms: int = 500,
    post_stimulus_ms: int = 1000,
    sample_rate_hz: int = 10_000,
    fade_out_ms: float = 15.0,
    stimulation_electrode_index: int = 0,
    electrode_count: int = 1,
) -> tuple[np.ndarray, dict[str, float | str]]:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if peak_current_ua < 0 or peak_current_ua > 25:
        raise ValueError("peak_current_ua must be in [0, 25] uA")
    expected = round(sample_rate_hz / 1000 * 1000)
    if len(waveform) != expected:
        raise ValueError(f"expected a one-second waveform with {expected} samples")
    if not 0 <= stimulation_electrode_index < electrode_count:
        raise ValueError("stimulation electrode index is out of range")
    tapered = apply_fade_out(waveform, sample_rate_hz, fade_out_ms)
    baseline_steps = round(baseline_ms * sample_rate_hz / 1000)
    stim_steps = len(tapered)
    post_steps = round(post_stimulus_ms * sample_rate_hz / 1000)
    current = np.zeros((baseline_steps + stim_steps + post_steps, electrode_count), dtype=np.float32)
    current[baseline_steps:baseline_steps + stim_steps, stimulation_electrode_index] = tapered * float(peak_current_ua)
    achieved = current[:, stimulation_electrode_index]
    stimulus_segment = achieved[baseline_steps:baseline_steps + stim_steps]
    if float(np.max(np.abs(achieved))) > float(peak_current_ua) + 1e-5:
        raise AssertionError("constructed current exceeds configured peak")
    digest = hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest()
    diagnostics = {
        "configured_peak_ua": float(peak_current_ua),
        "achieved_peak_ua": float(np.max(np.abs(achieved))),
        "achieved_rms_ua": float(np.sqrt(np.mean(achieved * achieved))),
        "final_5ms_rms_ua": float(np.sqrt(np.mean(stimulus_segment[-round(sample_rate_hz * 0.005):] ** 2))),
        "final_10ms_rms_ua": float(np.sqrt(np.mean(stimulus_segment[-round(sample_rate_hz * 0.010):] ** 2))),
        "stimulus_sha256": digest,
    }
    return current, diagnostics
