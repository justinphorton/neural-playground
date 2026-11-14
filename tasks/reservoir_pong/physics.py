"""Deterministic Pong trajectories and grid-entry events."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .records import BallSpec


@dataclass(frozen=True)
class CellEntry:
    onset_ms: float
    col: int
    row: int


def reflected_position(unfolded_x: float) -> float:
    """Reflect an unbounded horizontal coordinate into a unit-wide court."""
    folded = unfolded_x % 2.0
    return 1.0 - abs(folded - 1.0)


def trajectory_position(ball: BallSpec, fraction: float) -> tuple[float, float]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be inside [0, 1]")
    return reflected_position(ball.x0 + ball.vx * fraction), fraction


def landing_position(ball: BallSpec) -> float:
    return reflected_position(ball.x0 + ball.vx)


def _cell(x: float, y: float, cols: int, rows: int) -> tuple[int, int]:
    col = min(cols - 1, max(0, int(math.floor(x * cols))))
    row = min(rows - 1, max(0, int(math.floor(y * rows))))
    return col, row


def pulse_events(
    ball: BallSpec,
    cols: int,
    rows: int,
    quantize_ms: float | None = None,
) -> list[CellEntry]:
    """Return the initial occupied cell and each subsequent cell entry.

    A wall crossing alone never creates an event. Simultaneous horizontal and
    vertical crossings are combined into one diagonal cell entry.
    """
    if cols < 1 or rows < 1:
        raise ValueError("grid dimensions must be positive")
    if quantize_ms is not None and (not math.isfinite(quantize_ms) or quantize_ms <= 0):
        raise ValueError("quantize_ms must be positive and finite")

    crossings: list[float] = []
    end_x = ball.x0 + ball.vx
    lo, hi = sorted((ball.x0, end_x))
    first_k = math.floor(lo * cols) - 1
    last_k = math.ceil(hi * cols) + 1
    for k in range(first_k, last_k + 1):
        # Integer court edges are reflections, not transitions into a new cell.
        if k % cols == 0 or ball.vx == 0.0:
            continue
        fraction = (k / cols - ball.x0) / ball.vx
        if 1e-12 < fraction < 1.0 - 1e-12:
            crossings.append(fraction)
    for row in range(1, rows):
        fraction = row / rows
        if fraction < 1.0 - 1e-12:
            crossings.append(fraction)

    # Merge only mathematically simultaneous crossings. Distinct events that
    # quantize into the same stimulus tick are rejected by the spacing check.
    crossings.sort()
    merged: list[float] = []
    for fraction in crossings:
        if merged and abs(fraction - merged[-1]) <= 1e-12:
            continue
        merged.append(fraction)

    def at(fraction: float) -> tuple[int, int]:
        eps = min(1e-9, max(1e-12, 1e-10 / max(1.0, abs(ball.vx), rows)))
        sample = min(1.0, fraction + eps)
        x, y = trajectory_position(ball, sample)
        return _cell(x, y, cols, rows)

    entries = [CellEntry(0.0, *at(0.0))]
    previous = (entries[0].col, entries[0].row)
    for fraction in merged:
        current = at(fraction)
        if current == previous:
            continue
        onset = fraction * ball.fall_ms
        if quantize_ms is not None:
            onset = round(onset / quantize_ms) * quantize_ms
        entries.append(CellEntry(float(onset), *current))
        previous = current
    return entries


def validate_pulse_spacing(
    events: list[CellEntry], minimum_gap_ms: float, maximum_gap_ms: float
) -> None:
    if minimum_gap_ms <= 0 or maximum_gap_ms < minimum_gap_ms:
        raise ValueError("pulse-gap bounds are invalid")
    times = [event.onset_ms for event in events]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("pulse events collide after time quantization")
    for left, right in zip(times, times[1:]):
        gap = right - left
        if gap < minimum_gap_ms - 1e-9:
            raise ValueError(f"pulse gap {gap:.3f} ms is below minimum {minimum_gap_ms:.3f} ms")
        if gap > maximum_gap_ms + 1e-9:
            raise ValueError(f"pulse gap {gap:.3f} ms exceeds maximum {maximum_gap_ms:.3f} ms")
