"""Machine-readable metrics and compact figures for experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def write_metrics(metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=lambda value: value.tolist() if hasattr(value, "tolist") else float(value)) + "\n")


def plot_accuracy_curve(windows_ms: Sequence[tuple[float, float]], scores: Sequence[float], confidence_intervals: Sequence[tuple[float, float]], p_values: Sequence[float], path: Path, chance: float = 0.5) -> None:
    windows_ms = np.asarray(windows_ms, dtype=float)
    scores = np.asarray(scores, dtype=float)
    intervals = np.asarray(confidence_intervals, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    centers = windows_ms.mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, scores, marker="o", label="grouped-CV balanced accuracy")
    ax.fill_between(centers, intervals[:, 0], intervals[:, 1], alpha=0.2, label="95% grouped bootstrap CI")
    ax.axhline(chance, color="black", linestyle="--", linewidth=1, label="chance")
    significant = p_values < 0.05
    if np.any(significant):
        ax.scatter(centers[significant], scores[significant], color="crimson", zorder=3, label="Holm-significant")
    ax.set(xlabel="post-stimulus time (ms)", ylabel="balanced accuracy", ylim=(0, 1))
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stimulus_zero(current: np.ndarray, sample_rate_hz: int, path: Path) -> None:
    current = np.asarray(current)
    time_ms = np.arange(len(current)) * 1000.0 / sample_rate_hz
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(time_ms, current[:, 0] if current.ndim == 2 else current, linewidth=0.7)
    ax.axvline(500, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(1500, color="black", linestyle="--", linewidth=0.8)
    ax.set(xlabel="trial time (ms)", ylabel="current (uA)", title="stimulus-zero diagnostic")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def plot_amplitude_screen(summaries, path: Path) -> None:
    amplitudes = [summary.amplitude_ua for summary in summaries]
    during = [summary.during_firing_rate for summary in summaries]
    post = [summary.post_firing_rate for summary in summaries]
    early = [summary.early_score for summary in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(amplitudes, during, marker="o", label="during"); axes[0].plot(amplitudes, post, marker="o", label="post"); axes[0].set(xlabel="peak current (uA)", ylabel="firing rate"); axes[0].legend(frameon=False)
    axes[1].plot(amplitudes, early, marker="o"); axes[1].axhline(0.5, color="black", linestyle="--"); axes[1].set(xlabel="peak current (uA)", ylabel="early post balanced accuracy", ylim=(0, 1))
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)
