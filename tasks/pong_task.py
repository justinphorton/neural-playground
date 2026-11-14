"""Reservoir Pong: does the dish's response to a falling ball predict where it
lands better than the raw pulses do?

Same 9 sections as template_task.py. Game physics live in pong/game.py.

=============================================================== PLAN FIRST
TASK:        a ball falls through an 8x6 court at fixed downward speed with a
             random sideways speed, bouncing off the side walls. Predict the
             landing x. The paddle commits at half height, so the readout only
             sees the first half of the fall. Bounces after the commit make
             landing a nonlinear function of what was seen.
INPUT:       each time the ball enters a new grid cell, one biphasic pulse on
             that cell's electrode. 48 court electrodes. Nothing else encoded:
             no velocity, no paddle. Velocity has to come from pulse timing.
ENCODING:    court cell (col, row) -> electrode index. Pulses at the cell-entry
             times from the game. At vx_max=3, 30 ms of fall = 6-10 pulses,
             1-10 ms apart. (vx_max=12 gave ~25 pulses 1 ms apart, which
             saturated the dish and left no readable structure.)
DECODING:    spike counts per electrode per 1 ms bin in a 10 ms window that
             ends 4 ms after the last pulse before commit. (First try was a
             window AFTER the last pulse; it was empty, the dish is exhausted
             by then. See the note in settings.)
HYPOTHESIS:  the dish state at commit time carries the recent trajectory
             (which cells fired recently and in what order, via refractoriness
             and recovery). A linear readout on that state should predict
             landing better than the same readout on the raw pulse list,
             specifically on balls that bounce after commit. From
             memory_probe.py, the dish holds ~15 ms of history, so it should
             see the last ~8 pulses. Prediction: dish catch rate > 44% baseline
             (pong/baseline.py), with the gain on bouncing balls.
METRIC:      catch rate (|pred - landing| < one cell width) and mean abs error,
             overall and split by bounce / no-bounce after commit.
===========================================================================
"""

import os
import sys

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pong"))
import game  # noqa: E402


# ===========================================================================
# 1. SETTINGS
# ===========================================================================
N_CELLS = 300
TRIAL_MS = 80             # 30 ms of fall before commit, then blind + window, with slack
AMPLITUDE_UA = 40.0
PULSE_MS = 1.0            # each phase
BIPHASIC = True           # negative phase then positive, per the spec
PULSES_PER_CELL = 3       # zap each cell 3 times, 1 ms apart, as the ball enters it.
                          # One pulse barely spilled to neighbouring columns (0.3-spike
                          # direction signal). Three pulses drive the cells harder so the
                          # reload is deep enough to see: 2.3-spike direction signal.
PULSE_GAP_MS = 1.0
# Readout window. First attempt was "skip a blind phase after the last pulse,
# then count for 10 ms". That window was EMPTY: 20+ pulses in 30 ms exhaust
# the dish and it goes silent within 2 ms of the last pulse (0-5 spikes).
# The information is in how the dish RESPONDED to the last few pulses while
# tired from the earlier ones, so read that: a window ending just after the
# last pulse, reaching back over the last ~4 pulses.
PRE_MS = 6.0              # (end-of-fall window, kept for comparison)
POST_MS = 4.0
BIN_MS = 1
FALL_BIN_MS = 3           # whole-fall readout: counts per electrode per 3 ms bin over 0..36 ms.
                          # The direction signal is ~1-2 spikes per zap; reading every zap's
                          # response gives the readout 6-10 copies of it instead of one.
N_BALLS = 400             # 400 balls ~ 10 min of simulation
COMMIT_HEIGHT = game.COMMIT_HEIGHT
RIDGE_ALPHA = 10.0
DATA_DIR = os.path.dirname(HERE)


# ===========================================================================
# 2. TASK DEFINITION
# ===========================================================================
# Inputs are balls, not a fixed list. Each ball is a dict from pong/game.py.
# target() is the landing x in [0, 1]. This is regression, not yes/no.
def make_balls(n=N_BALLS, seed=0):
    return game.make_dataset(n, COMMIT_HEIGHT, seed=seed)


def target(ball_record):
    return ball_record["landing"]


# ===========================================================================
# 3. DATA GENERATION
# ===========================================================================
def make_dataset(seed=0):
    return make_balls(seed=seed)


# ===========================================================================
# 4. ENCODING  (court cell -> electrode pulse at the time the ball enters it)
# ===========================================================================
class CourtEncoding(Encoding):
    court_electrodes: list[int]   # 48 electrode ids, one per court cell
    amplitude_ua: float = AMPLITUDE_UA
    dt: float = 0.1

    def pulses(self, ball_record):
        """(time_ms, electrode) for each cell the ball entered before commit,
        PULSES_PER_CELL times per cell."""
        out = []
        for t, c in ball_record["pulses"]:
            for k in range(PULSES_PER_CELL):
                out.append((t + k * PULSE_GAP_MS, self.court_electrodes[c]))
        return out

    def __call__(self, env, t_end, inputs):
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)
        half = int(PULSE_MS / self.dt)
        for t, ch in self.pulses(inputs):
            start = int(t / self.dt)
            if BIPHASIC:
                electrodes[start:start + half, ch] = -self.amplitude_ua
                electrodes[start + half:start + 2 * half, ch] = self.amplitude_ua
            else:
                electrodes[start:start + half, ch] = self.amplitude_ua
        return env.cell_stimulus(electrodes, dt=self.dt)


# ===========================================================================
# 5. DECODING  (skip the blind phase after the last pulse, then count)
# ===========================================================================
class PostCommitCounts(Decoding):
    duration: int = TRIAL_MS
    pre_ms: float = PRE_MS
    post_ms: float = POST_MS
    bin_ms: int = BIN_MS

    def setup(self, env):
        env.record_spikes()

    def features_from_channels(self, times_per_channel, env, last_pulse_ms):
        """End-of-fall window: last few pulses' response."""
        start = last_pulse_ms - self.pre_ms
        end = last_pulse_ms + self.post_ms
        edges = np.arange(start, end + self.bin_ms, self.bin_ms)
        counts = [np.histogram(times_per_channel.get(ch, []), bins=edges)[0]
                  for ch in env.io.channel_ids]
        return np.concatenate(counts)

    def whole_fall_features(self, times_per_channel, env, court_electrodes):
        """Whole-fall window: counts per COURT electrode per 3 ms bin, 0..36 ms.
        Only the 48 court electrodes, since the others hear almost nothing."""
        edges = np.arange(0, 36 + FALL_BIN_MS, FALL_BIN_MS)
        counts = [np.histogram(times_per_channel.get(ch, []), bins=edges)[0]
                  for ch in court_electrodes]
        return np.concatenate(counts)

    def __call__(self, signal, env=None):
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)
        return {
            "times_per_channel": {int(ch): np.asarray(t, dtype=float)
                                  for ch, t in times_per_channel.items()},
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
    return env


def court_electrodes(env):
    """Map the 8x6 court onto a contiguous 8x6 block of the electrode grid so
    adjacent court cells are adjacent electrodes (200 um apart). The chip is
    8 wide x 16 tall; pick the 6 rows with the most neurons under them.

    First version used the 48 strongest electrodes anywhere on the chip, which
    broke adjacency: the ball's neighbours in the court were not neighbours on
    the chip, so the tiredness pattern did not follow the ball's path.
    """
    reach = np.asarray(env.channel_reach()).sum(axis=1)
    xy = np.asarray(env.io.electrode_coordinates)[:, 1:3]
    xs = np.unique(np.round(xy[:, 0], 1))
    ys = np.unique(np.round(xy[:, 1], 1))
    assert len(xs) >= game.GRID_W and len(ys) >= game.GRID_H
    best = None
    for r0 in range(len(ys) - game.GRID_H + 1):
        for c0 in range(len(xs) - game.GRID_W + 1):
            block = []
            for r in range(game.GRID_H):
                for c in range(game.GRID_W):
                    e = int(np.argmin(np.hypot(xy[:, 0] - xs[c0 + c], xy[:, 1] - ys[r0 + r])))
                    block.append(e)
            tot = reach[block].sum()
            if best is None or tot > best[0]:
                best = (tot, block)
    return best[1]


def make_pipeline(env):
    return CourtEncoding(court_electrodes=court_electrodes(env)), PostCommitCounts()


# ===========================================================================
# 7. COLLECT NEURAL FEATURES
# ===========================================================================
def collect_trials(env, encoding, decoding, dataset):
    features, labels, meta = [], [], []
    for i, ball in enumerate(dataset):
        if not ball["pulses"]:
            continue
        env.clear()
        out = env(decoding, inputs=ball, encoding=encoding)
        last_pulse = ball["pulses"][-1][0]
        features.append(np.concatenate([
            decoding.features_from_channels(out["times_per_channel"], env, last_pulse),
            decoding.whole_fall_features(out["times_per_channel"], env, encoding.court_electrodes)]))
        labels.append(target(ball))
        meta.append({"bounces_after": ball["bounces_after"], "pulses": ball["pulses"]})
        print(f"  ball {i + 1}/{len(dataset)}", end="\r")
    print()
    return np.array(features), np.array(labels), meta


# ===========================================================================
# 8. READOUT + EVALUATION
# ===========================================================================
def evaluate(X, y, meta, name, seeds=(0, 1, 2)):
    bounce = np.array([m["bounces_after"] > 0 for m in meta])
    half = 1.0 / game.GRID_W
    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te, b_tr, b_te = train_test_split(
            X, y, bounce, test_size=0.3, random_state=seed)
        pred = Ridge(alpha=RIDGE_ALPHA).fit(X_tr, y_tr).predict(X_te)
        err = np.abs(pred - y_te)
        caught = err < half
        rows.append((err.mean(), caught.mean(),
                     caught[b_te].mean() if b_te.any() else np.nan,
                     caught[~b_te].mean() if (~b_te).any() else np.nan))
    r = np.nanmean(rows, axis=0)
    print(f"{name:<34} err {r[0]:.3f}  catch {r[1]:.0%}  "
          f"catch bounce {r[2]:.0%}  catch no-bounce {r[3]:.0%}")
    return r


def baseline_features(meta):
    return np.array([game.pulses_to_compact(m["pulses"]) for m in meta])


# ===========================================================================
# 9. MAIN
# ===========================================================================
def main():
    env = build_reservoir()
    encoding, decoding = make_pipeline(env)
    dataset = make_dataset()
    print(f"Simulating {len(dataset)} balls, commit at height {COMMIT_HEIGHT}, "
          f"court on {len(encoding.court_electrodes)} electrodes...")
    X_dish, y, meta = collect_trials(env, encoding, decoding, dataset)
    X_raw = baseline_features(meta)

    n_end = len(env.io.channel_ids) * int((PRE_MS + POST_MS) / BIN_MS)
    X_end, X_fall = X_dish[:, :n_end], X_dish[:, n_end:]
    print()
    evaluate(X_raw, y, meta, "BASELINE  ridge on last-6 pulses (compact)")
    evaluate(X_end, y, meta, "DISH      end-of-fall window only")
    evaluate(X_fall, y, meta, "DISH      whole-fall, court electrodes")
    evaluate(np.hstack([X_raw, X_fall]), y, meta, "BOTH      compact + whole-fall")
    rng = np.random.default_rng(0)
    evaluate(X_fall, rng.permutation(y), meta, "CONTROL   whole-fall, shuffled labels")
    env.close()


PLAYGROUND = {
    "tagline": "A ball falls and bounces. The dish sees the first half. Can a line read where it lands?",
    "input_names": ["ball"],
}


if __name__ == "__main__":
    main()
