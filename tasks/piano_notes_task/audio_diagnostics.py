"""Report source-level RMS imbalance without feeding audio into neural X."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .audio_processing import AudioAudit
from .dataset_manifest import RecordingManifestRow


def audio_rms_diagnostic(rows: Sequence[RecordingManifestRow], audits: Sequence[AudioAudit]) -> dict:
    if len(rows) != len(audits):
        raise ValueError("manifest and audits must have equal length")
    by_note = {
        note: np.asarray([audit.normalized_rms for row, audit in zip(rows, audits) if row.note == note], dtype=float)
        for note in ("C4", "E4")
    }
    if any(len(values) == 0 for values in by_note.values()):
        raise ValueError("both notes are required")
    c4, e4 = by_note["C4"], by_note["E4"]
    pooled = np.concatenate([c4, e4])
    # This is a nuisance diagnostic, not evidence for neural memory.
    return {
        "c4_count": int(len(c4)), "e4_count": int(len(e4)),
        "c4_normalized_rms_mean": float(c4.mean()), "e4_normalized_rms_mean": float(e4.mean()),
        "c4_normalized_rms_std": float(c4.std(ddof=1)) if len(c4) > 1 else 0.0,
        "e4_normalized_rms_std": float(e4.std(ddof=1)) if len(e4) > 1 else 0.0,
        "pooled_rms_range": [float(pooled.min()), float(pooled.max())],
        "mean_difference_c4_minus_e4": float(c4.mean() - e4.mean()),
        "interpretation": "audio-only nuisance diagnostic; it is not neural classifier accuracy",
    }
