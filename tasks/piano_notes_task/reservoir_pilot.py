"""Shared electrode-selection helpers for the fixed-reservoir experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


N_CELLS = 300
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]


@dataclass(frozen=True)
class ElectrodeSite:
    stimulus_index: int  # column index in the MEA current matrix
    stimulus_id: int  # physical channel ID
    neighbor_ids: tuple[int, ...]  # four adjacent and four diagonal IDs


def choose_site(electrodes: np.ndarray, reach: np.ndarray) -> ElectrodeSite:
    """Choose the highest-reach interior electrode with all eight grid neighbors."""
    electrodes = np.asarray(electrodes)
    reach = np.asarray(reach)
    if electrodes.ndim != 2 or electrodes.shape[1] < 3 or reach.shape[0] != len(electrodes):
        raise ValueError("Electrode coordinates and reach must have matching rows")

    xs = np.unique(electrodes[:, 1])
    ys = np.unique(electrodes[:, 2])
    locations = {
        (float(row[1]), float(row[2])): (index, int(row[0]))
        for index, row in enumerate(electrodes)
    }
    scores = reach.sum(axis=1)
    candidates = []
    for index, row in enumerate(electrodes):
        x, y = row[1], row[2]
        xi, yi = np.searchsorted(xs, x), np.searchsorted(ys, y)
        if not (0 < xi < len(xs) - 1 and 0 < yi < len(ys) - 1):
            continue
        neighbor_positions = [
            (float(xs[xi + dx]), float(ys[yi + dy]))
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        ]
        if all(position in locations for position in neighbor_positions):
            neighbor_ids = tuple(locations[position][1] for position in neighbor_positions)
            candidates.append((float(scores[index]), ElectrodeSite(index, int(row[0]), neighbor_ids)))
    if not candidates:
        raise ValueError("No electrode has all eight neighbors in this layout")
    return max(candidates, key=lambda pair: pair[0])[1]


