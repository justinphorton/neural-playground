"""Reservoir Pong: predict which side a bouncing ball lands on from the network's response.

Same 9 sections, in the same order, as template_task.py. Run tasks/pong_game.py
first: it picks the difficulty (no neurons, seconds) and this file imports its game.

=============================================================== PLAN
TASK:        balls fall through a box, bouncing off the side walls. After seeing the
             first N_ROWS rows of the fall, which half of the bottom will the ball land on?
INPUT:       the column the ball is in at each observed row, e.g. (1, 2, 3).
             Only the column patterns a real ball can produce are used (20 for 4x3).
             Each pattern's label is the side MOST of its balls land on (pong_game.py
             simulates them), so the label is fixed per pattern, like XOR.
ENCODING:    one 1 ms pulse per row, at fixed times ROW_GAP_MS apart, on the electrode for
             that row's column. Height = timing, sideways position = which electrode.
             Columns map to electrodes in left-to-right order on the array.
             All pulses fit inside the ~15-20 ms the network remembers (memory_task.py).
DECODING:    spike counts per electrode in 5 ms bins over the whole trial, with the first
             BLANK_MS after each pulse thrown away (a real rig can't record then).
HYPOTHESIS:  landing side depends on the walls (a fold), so a line on the raw columns
             can't get it. If pulses arriving within a few ms interact in the network, a
             linear readout on the spikes should beat the same readout on the raw input.
METRIC:      test accuracy: reservoir vs. linear on raw columns vs. linear on one-hot
             slots; plus accuracy after each pulse, and expected in-game accuracy.
===========================================================================
"""

import os

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

import pong_game as game


# ===========================================================================
# 1. SETTINGS
# ===========================================================================
N_CELLS = 300  # neurons in the reservoir
# Synapse strengths applied on top of livn's model defaults ({} = defaults). The defaults give
# NMDA weight 0 and very few connections inside a small patch, so activity dies within ~2 ms.
WEIGHTS = {}  # e.g. {"EXC_EXC-dend-AMPA-weight": 35.0, "EXC_EXC-dend-NMDA-weight": 15.0} (defaults 7 and 0)
ELECTRODES_PER_COLUMN = 1  # 2 = each column fires a vertical pair of electrodes (more neurons, like demo_xor)
N_COLS = 4  # columns across the court = stimulating electrodes (check how many reach the patch!)
N_ROWS = 3  # observed rows = pulses per ball
DIFFICULTY = 4  # rows the ball falls after the commit point (pong_game.py: biggest gap at 4 for 4x3)
VMAX = 1.0  # max sideways speed, cells per row

FIRST_PULSE_MS = 20.0  # first row's pulse (leaves 20 ms of background before it)
ROW_GAP_MS = 5.0  # time between rows; XOR mixed best at 6 ms, memory fades by ~15-20 ms
WINDOW_MS = 30.0  # keep recording this long after the last pulse
TRIAL_MS = int(FIRST_PULSE_MS + ROW_GAP_MS * (N_ROWS - 1) + WINDOW_MS)  # 60 ms for 3 rows

AMPLITUDE_UA = 40.0
BIN_MS = 5
# Recording blind after each pulse. A real rig needs ~2 ms (Loeffler-style blanking), BUT in this
# simulated culture ~96% of spikes come within 2 ms of a pulse (cells fired directly by the electrode;
# almost no network echo). With BLANK_MS = 2 the readout sees ~3 spikes per ball and fails (38%).
# So 0 here, matching demo_xor.py and memory_task.py; set 2 to see the realistic case.
BLANK_MS = 0.0
TRIALS_PER_INPUT = 50  # repeats per column pattern: 20 patterns x 50 = 1,000 trials (~17 min)

JITTER_MS = 0.0  # training variation, as in demo_xor.py (it uses 1.5 and (30, 50))
AMPLITUDE_RANGE_UA = None

RANDOM_STATE = 0
N_CONTROL = 30  # repeats of each control (shuffled labels, random labellings)
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo folder
SAVE_PATH = os.path.join(DATA_DIR, "pong_trials.npz")  # recorded trials, reused by pong_landing.py

ROW_TIMES_MS = [FIRST_PULSE_MS + r * ROW_GAP_MS for r in range(N_ROWS)]


# ===========================================================================
# 2. TASK DEFINITION
# ---------------------------------------------------------------------------
# The game (pong_game.py) simulates many balls. Each one produces a column
# pattern; we keep every pattern that appears, and label it with the side most
# of its balls land on. PURITY[x] = fraction of its balls that land on that side
# (1.0 = always; 0.5 = coin flip even for a perfect player).
# ===========================================================================
def _game_patterns(n_balls=100_000, seed=0):
    game.N_COLS, game.N_ROWS, game.VMAX = N_COLS, N_ROWS, VMAX
    rng = np.random.default_rng(seed)
    x0, vx = game.sample_balls(n_balls, rng)
    cols = game.observe(x0, vx)
    side = game.lands_right(game.landing(x0, vx, DIFFICULTY))
    patterns, inverse, counts = np.unique(cols, axis=0, return_inverse=True, return_counts=True)
    p_right = np.bincount(inverse.ravel(), weights=side) / counts
    inputs = [tuple(int(c) for c in p) for p in patterns]
    labels = {x: int(p > 0.5) for x, p in zip(inputs, p_right)}
    purity = {x: float(max(p, 1 - p)) for x, p in zip(inputs, p_right)}
    freq = {x: float(c / n_balls) for x, c in zip(inputs, counts)}
    return inputs, labels, purity, freq


INPUTS, LABELS, PURITY, FREQUENCY = _game_patterns()


def target(x):
    return LABELS[tuple(x)]  # 1 = lands on the right half


# ===========================================================================
# 3. DATA GENERATION
# ===========================================================================
def make_dataset():
    return [x for _ in range(TRIALS_PER_INPUT) for x in INPUTS]


# ===========================================================================
# 4. ENCODING  (column pattern -> one pulse per row)
# ===========================================================================
class CourtEncoding(Encoding):
    """Row r of the fall -> a pulse at ROW_TIMES_MS[r] on the electrode for its column."""

    channels: list[list[int]]  # electrode(s) for each column, left to right
    amplitude_ua: float = AMPLITUDE_UA
    vary: bool = False
    dt: float = 0.1

    def pulses(self, inputs):
        return [(t, ch) for t, c in zip(ROW_TIMES_MS, inputs) for ch in self.channels[c]]

    def __call__(self, env, t_end, inputs):
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)
        nudges = {}  # electrodes pulsed at the same time share one nudge, so pairs stay together
        for t, ch in self.pulses(inputs):
            if t not in nudges:
                shift = np.random.uniform(-JITTER_MS, JITTER_MS) if self.vary else 0.0
                amp = (np.random.uniform(*AMPLITUDE_RANGE_UA)
                       if self.vary and AMPLITUDE_RANGE_UA else self.amplitude_ua)
                nudges[t] = (shift, amp)
            shift, amplitude = nudges[t]
            t = t + shift
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = amplitude
        return env.cell_stimulus(electrodes, dt=self.dt)


# ===========================================================================
# 5. DECODING  (spikes -> counts per electrode per 5 ms, blind after each pulse)
# ---------------------------------------------------------------------------
# Every ball has a pulse at every row time (only the electrode changes), so the
# blind windows are the same for every input and can't leak the answer.
# ===========================================================================
class BlankedChannelCounts(Decoding):
    duration: int = TRIAL_MS
    bin_ms: int = BIN_MS
    blank_ms: float = BLANK_MS

    def setup(self, env):
        env.record_spikes()

    def keep(self, times):
        """Drop spikes inside any post-pulse blind window (widened by the training jitter)."""
        times = np.asarray(times, dtype=float)
        if self.blank_ms <= 0 or times.size == 0:
            return times
        blind = np.zeros(times.shape, dtype=bool)
        for p in ROW_TIMES_MS:
            blind |= (times >= p - JITTER_MS) & (times < p + JITTER_MS + max(self.blank_ms, 1.0))
        return times[~blind]

    def __call__(self, signal, env=None):
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)
        edges = np.arange(0, self.duration + self.bin_ms, self.bin_ms)
        counts = [np.histogram(self.keep(times_per_channel.get(ch, [])), bins=edges)[0]
                  for ch in env.io.channel_ids]
        return {
            "features": np.concatenate(counts),  # electrode-major: 128 x (TRIAL_MS / BIN_MS)
            "spike_ids": np.asarray(signal.spike_ids),
            "spike_times": np.asarray(signal.spike_times),
        }


# ===========================================================================
# 6. RESERVOIR
# ===========================================================================
def build_reservoir():
    env = Env(predefined("EI", download_directory=DATA_DIR))
    env.selection(N_CELLS, method="patch")
    env.init().apply_model_defaults()
    if WEIGHTS:
        env.set_weights(WEIGHTS)
    return env


def strongest_electrodes(env, n=1):
    """The n electrodes that couple most strongly into our patch of cells."""
    reach = np.asarray(env.channel_reach()).sum(axis=1)
    return [int(ch) for ch in np.argsort(reach)[::-1][:n]]


def court_electrodes(env):
    """N_COLS neighbouring electrodes in one straight row of the array, left to right, so
    neighbouring columns sit on neighbouring electrodes. Of all such runs, picks the one whose
    weakest electrode reaches the patch best. (The strongest electrodes overall form a 2x2
    square, not a line.) Prints their reach so you can see if the weakest is too weak."""
    reach = np.asarray(env.channel_reach()).sum(axis=1)
    xy = np.asarray(env.io.electrode_coordinates)[:, 1:3]
    ys = np.unique(xy[:, 1])
    at = {(float(xy[ch][0]), float(xy[ch][1])): int(ch) for ch in range(len(xy))}
    best, best_score = None, -1.0
    for yi, y in enumerate(ys):
        # with pairs, each column = this row's electrode + the one directly below it
        rows = [y] if ELECTRODES_PER_COLUMN == 1 else ([y, ys[yi + 1]] if yi + 1 < len(ys) else None)
        if rows is None:
            continue
        line = sorted(np.flatnonzero(xy[:, 1] == y), key=lambda ch: xy[ch][0])
        for i in range(len(line) - N_COLS + 1):
            run = line[i:i + N_COLS]
            gaps = np.diff([xy[ch][0] for ch in run])
            if not np.allclose(gaps, gaps[0]):
                continue
            cols = [[at.get((float(xy[ch][0]), float(r))) for r in rows] for ch in run]
            if any(e is None for c in cols for e in c):
                continue
            score = min(reach[e] for c in cols for e in c)
            if score > best_score:
                best, best_score = cols, float(score)
    print(f"Court electrodes per column (left to right): {best}; weakest reach = {best_score / reach.max():.0%} of the strongest")
    return best


def make_pipeline(env):
    encoding = CourtEncoding(channels=court_electrodes(env),
                             vary=JITTER_MS > 0 or AMPLITUDE_RANGE_UA is not None)
    return encoding, BlankedChannelCounts()


# ===========================================================================
# 7. COLLECT NEURAL FEATURES
# ===========================================================================
def collect_trials(env, encoding, decoding, dataset):
    features, labels = [], []
    for i, x in enumerate(dataset):
        env.clear()  # each ball starts from a fresh network: no memory of the last ball
        out = env(decoding, inputs=x, encoding=encoding)
        features.append(out["features"])
        labels.append(target(x))
        print(f"  ball {i + 1}/{len(dataset)}", end="\r")
    print()
    return np.array(features), np.array(labels)


# ===========================================================================
# 8. READOUT + EVALUATION
# ---------------------------------------------------------------------------
# Every model is a linear readout (ridge, regularisation picked by
# cross-validation on the training set only) thresholded at 0.5. Only the
# features change. The split keeps every column pattern in both halves.
# ===========================================================================
def raw_features(inputs):
    return np.asarray(inputs, dtype=float)  # the column at each row


def onehot_features(inputs):
    return game.onehot_features(np.asarray(inputs, dtype=int))  # one slot per (row, column)


def random_features(inputs, n=500, seed=RANDOM_STATE):
    """A noise-free random nonlinear expansion of the one-hot input: what an ideal,
    perfectly repeatable 'reservoir' with no biology would give. Upper reference."""
    rng = np.random.default_rng(seed)
    X = onehot_features(inputs)
    W, b = rng.normal(size=(X.shape[1], n)), rng.normal(size=n)
    return np.tanh(X @ W + b)


def row_prefix(X, rows_seen):
    """Reservoir features up to just before the next row's pulse (or the trial end):
    what the readout could use if it had to answer after `rows_seen` pulses."""
    n_bins = TRIAL_MS // BIN_MS
    end_ms = ROW_TIMES_MS[rows_seen] if rows_seen < N_ROWS else TRIAL_MS
    keep = int(np.ceil(end_ms / BIN_MS))
    return X.reshape(len(X), -1, n_bins)[:, :, :keep].reshape(len(X), -1)


def fit_score(X, y, train, test):
    readout = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[train], y[train])
    pred = (readout.predict(X[test]) > 0.5).astype(int)
    return pred, np.mean(pred == y[test])


def game_accuracy(inputs_test, correct):
    """Expected share of REAL balls caught on the right side: weight each pattern by how
    often it occurs, and account for patterns whose balls don't all land on one side."""
    total = 0.0
    for x in INPUTS:
        mask = np.array([tuple(t) == x for t in inputs_test])
        if mask.any():
            a = correct[mask].mean()
            total += FREQUENCY[x] * (a * PURITY[x] + (1 - a) * (1 - PURITY[x]))
    return total / sum(FREQUENCY[x] for x in INPUTS)


def evaluate(X_reservoir, y, inputs):
    inputs = [tuple(x) for x in inputs]
    pattern = np.array([INPUTS.index(x) for x in inputs])
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=0.3, stratify=pattern, random_state=RANDOM_STATE)
    inputs_test = [inputs[i] for i in test]
    ceiling = sum(FREQUENCY[x] * PURITY[x] for x in INPUTS)

    models = {
        "linear on raw columns": raw_features(inputs),
        "linear on one-hot slots": onehot_features(inputs),
        "RESERVOIR (linear on spikes)": X_reservoir,
        "random nonlinear features (ref)": random_features(inputs),
    }
    print(f"\n{'readout':<34} {'test acc':>8} {'game acc':>9}")
    print(f"{'(best possible)':<34} {'100%':>8} {ceiling:>9.0%}")
    preds = {}
    for name, X in models.items():
        pred, acc = fit_score(X, y, train, test)
        preds[name] = pred
        print(f"{name:<34} {acc:>8.0%} {game_accuracy(inputs_test, pred == y[test]):>9.0%}")

    # Controls. Labels are fixed per pattern, so there are effectively only 20 test cases and a
    # single run swings by ~10 points: repeat each control N_CONTROL times.
    rng = np.random.default_rng(RANDOM_STATE)
    real = np.mean(preds["RESERVOIR (linear on spikes)"] == y[test])
    shuffled = []
    for _ in range(N_CONTROL):
        y_s = y.copy()
        y_s[train] = rng.permutation(y[train])
        shuffled.append(np.mean(fit_score(X_reservoir, y_s, train, test)[0] == y[test]))
    print(f"{'control: reservoir, shuffled labels':<34} {np.mean(shuffled):>8.0%}   "
          f"(should be ~50%; 95th pct {np.percentile(shuffled, 95):.0%}; real beats {np.mean(np.array(shuffled) < real):.0%} of runs)")

    # Is the Pong labelling harder than a RANDOM labelling of the same patterns? If the walls make it
    # hard for a line, the baseline should score far below its random-labelling average.
    labs = np.array([LABELS[x] for x in INPUTS])
    pat = np.array([INPUTS.index(x) for x in inputs])
    for name, X in (("linear on one-hot slots", models["linear on one-hot slots"]),
                    ("RESERVOIR (linear on spikes)", X_reservoir)):
        scores = []
        for _ in range(N_CONTROL):
            y_r = rng.permutation(labs)[pat]
            scores.append(np.mean(fit_score(X, y_r, train, test)[0] == y_r[test]))
        pong = np.mean(preds[name] == y[test])
        print(f"  {name}: Pong labels {pong:.0%}, random labellings {np.mean(scores):.0%} on average")

    print("\nAccuracy if the readout had to answer after k pulses (k = rows seen):")
    print(f"{'k':>3} {'one-hot (first k rows)':>23} {'reservoir':>10}")
    for k in range(1, N_ROWS + 1):
        _, acc_base = fit_score(onehot_features([x[:k] + (0,) * (N_ROWS - k) for x in inputs]), y, train, test)
        _, acc_res = fit_score(row_prefix(X_reservoir, k), y, train, test)
        print(f"{k:>3} {acc_base:>23.0%} {acc_res:>10.0%}")

    print("\nPer pattern (test trials): pattern -> label, purity, one-hot acc, reservoir acc")
    base, res = preds["linear on one-hot slots"] == y[test], preds["RESERVOIR (linear on spikes)"] == y[test]
    for x in INPUTS:
        mask = np.array([t == x for t in inputs_test])
        print(f"  {x} -> {'R' if LABELS[x] else 'L'}  purity {PURITY[x]:4.0%}   "
              f"one-hot {base[mask].mean():4.0%}   reservoir {res[mask].mean():4.0%}")


# ===========================================================================
# 9. MAIN
# ===========================================================================
def main():
    print(f"Pong: {N_COLS} columns, {N_ROWS} rows (pulses at {ROW_TIMES_MS} ms), difficulty {DIFFICULTY}, "
          f"{len(INPUTS)} column patterns, blind {BLANK_MS} ms after each pulse")
    env = build_reservoir()
    encoding, decoding = make_pipeline(env)
    dataset = make_dataset()
    print(f"Running {len(dataset)} balls...")
    X_reservoir, y = collect_trials(env, encoding, decoding, dataset)
    np.savez_compressed(SAVE_PATH, X=X_reservoir, y=y, inputs=np.asarray(dataset),
                        channels=np.asarray(encoding.channels), blank_ms=BLANK_MS, row_gap_ms=ROW_GAP_MS)
    print(f"Saved the trials to {SAVE_PATH} (pong_landing.py reuses them)")
    evaluate(X_reservoir, y, dataset)
    env.close()


# ===========================================================================
# PLAYGROUND TEXT (optional)
# ===========================================================================
PLAYGROUND = {
    "tagline": "Where will the ball land? The walls make it a fold no straight line can follow. "
               "Can a fixed network of neurons unfold it for a linear readout?",
    "encoding": f"Each ball becomes {N_ROWS} pulses, one per row of its fall, {ROW_GAP_MS:g} ms apart. "
                f"<b>Which</b> electrode fires says which of the {N_COLS} columns the ball is in; <b>when</b> it "
                "fires says how far down it is. Input (1, 2, 3) = a ball drifting right.",
    "howto": "<b>Pong by hand:</b> pulse the column electrodes left to right at 20, 25, 30 ms (a ball drifting right, "
             "about to hit the right wall). Then reverse the order.",
    "baseline": "A line on the columns extrapolates straight ahead and never bounces. A ball drifting right fast "
                "enough lands on the <i>left</i>: the walls fold the court, and a line can't fold.",
    "ideas": [
        "<b>Change the row gap</b> (ROW_GAP_MS) to 3, 10 and 20 ms. Past the network's ~15-20 ms memory, the rows "
        "stop interacting and the reservoir should fall back to the baseline.",
        "<b>Blind the recording longer</b> (BLANK_MS = 10). Real rigs lose a few ms after each pulse; how much "
        "of the useful response survives?",
        "<b>Make it harder</b> (DIFFICULTY in pong_game.py) and watch the gap between baseline and best possible.",
    ],
}


if __name__ == "__main__":
    main()
