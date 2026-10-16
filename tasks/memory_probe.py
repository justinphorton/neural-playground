"""Hidden memory: does a PROBE zap respond differently if a zap came before it?

Follow-up to memory_curve.py. That experiment found the spikes from a zap are
gone within 5 ms. But XOR works at a 6 ms gap, so the dish still "remembers" at
6 ms, just not as extra firing. It remembers as tired cells that CAN'T fire.
An empty window can't show tiredness. A second zap can: it hits the tired cells
and comes out smaller.

=============================================================== PLAN FIRST
TASK:        "did a zap happen at 20 ms?"  (1 = yes, 0 = no)
INPUT:       one bit: (1,) = conditioning zap at 20 ms, (0,) = none.
             EITHER WAY, a probe zap is ALWAYS delivered at 20 + PROBE_DELAY_MS.
ENCODING:    conditioning zap (if bit = 1) and probe zap: each 1 ms, 40 uA on
             electrodes 3 and 51. Same as XOR's pulses.
DECODING:    spike counts per electrode per 1 ms bin, but ONLY in the 8 ms
             after the probe. The readout never sees the conditioning zap's burst,
             only the probe's response.
HYPOTHESIS:  if the conditioning zap happened, the probe hits tired cells and its
             burst is smaller / from different electrodes. The readout can tell.
             This should last LONGER than the 5 ms from memory_curve.py, because
             tiredness outlasts firing. Expect above chance out to ~10-15 ms,
             since XOR worked at 6 and failed at 30.
METRIC:      accuracy vs PROBE_DELAY_MS, plotted next to the memory_curve result.
             The gap between the two curves is the memory that lives in "can't fire".
===========================================================================
"""

import os

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split


# ===========================================================================
# 1. SETTINGS
# ===========================================================================
N_CELLS = 300
TRIAL_MS = 80
COND_MS = 20.0  # the conditioning zap (if the input is 1)
PROBE_DELAY_MS = 6.0  # the probe comes this long after it  <-- the knob we sweep
AMPLITUDE_UA = 40.0
PROBE_WINDOW_MS = 8  # decode only this long after the probe
BIN_MS = 1  # fine bins: the probe burst is short, and its shape matters
TRIALS_PER_INPUT = 100
RIDGE_ALPHA = 10.0  # ridge instead of plain least squares: 1,024 features, 200 trials

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 2. TASK DEFINITION
# ===========================================================================
INPUTS = [(0,), (1,)]


def target(x):
    return x[0]  # "did the conditioning zap happen?"


# ===========================================================================
# 3. DATA GENERATION
# ===========================================================================
def make_dataset():
    return [x for _ in range(TRIALS_PER_INPUT) for x in INPUTS]


# ===========================================================================
# 4. ENCODING  (conditioning zap if bit = 1; probe zap ALWAYS)
# ===========================================================================
class ProbeEncoding(Encoding):
    channels: list[int]
    amplitude_ua: float = AMPLITUDE_UA
    probe_delay_ms: float = PROBE_DELAY_MS
    dt: float = 0.1

    def pulses(self, inputs):
        out = []
        if inputs[0] == 1:
            out += [(COND_MS, ch) for ch in self.channels]
        out += [(COND_MS + self.probe_delay_ms, ch) for ch in self.channels]
        return out

    def __call__(self, env, t_end, inputs):
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)
        for t, ch in self.pulses(inputs):
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = self.amplitude_ua
        return env.cell_stimulus(electrodes, dt=self.dt)


# ===========================================================================
# 5. DECODING  (only the probe's response)
# ===========================================================================
class ProbeResponseCounts(Decoding):
    duration: int = TRIAL_MS
    probe_delay_ms: float = PROBE_DELAY_MS
    window_ms: int = PROBE_WINDOW_MS
    bin_ms: int = BIN_MS

    def setup(self, env):
        env.record_spikes()

    def features_from_channels(self, times_per_channel, env):
        start = COND_MS + self.probe_delay_ms
        edges = np.arange(start, start + self.window_ms + self.bin_ms, self.bin_ms)
        counts = [np.histogram(times_per_channel.get(ch, []), bins=edges)[0] for ch in env.io.channel_ids]
        return np.concatenate(counts)

    def __call__(self, signal, env=None):
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)
        return {
            "features": self.features_from_channels(times_per_channel, env),
            "times_per_channel": {int(ch): np.asarray(t, dtype=float) for ch, t in times_per_channel.items()},
            "spike_ids": np.asarray(signal.spike_ids),
            "spike_times": np.asarray(signal.spike_times),
        }


# ===========================================================================
# 6. RESERVOIR  (same as XOR)
# ===========================================================================
def build_reservoir():
    env = Env(predefined("EI", download_directory=DATA_DIR))
    env.selection(N_CELLS, method="patch")
    env.init().apply_model_defaults()
    return env


def strongest_electrodes(env, n=1):
    reach = np.asarray(env.channel_reach()).sum(axis=1)
    return [int(ch) for ch in np.argsort(reach)[::-1][:n]]


def neighbour_electrode(env, ch):
    xy = np.asarray(env.io.electrode_coordinates)[:, 1:3]
    candidates = [e for e in strongest_electrodes(env, n=4) if e != ch]
    return min(candidates, key=lambda e: np.linalg.norm(xy[e] - xy[ch]))


def make_pipeline(env, probe_delay_ms=PROBE_DELAY_MS):
    main = strongest_electrodes(env)[0]
    channels = [main, neighbour_electrode(env, main)]
    return ProbeEncoding(channels=channels, probe_delay_ms=probe_delay_ms), ProbeResponseCounts(probe_delay_ms=probe_delay_ms)


# ===========================================================================
# 7. COLLECT NEURAL FEATURES
# ===========================================================================
def collect_trials(env, encoding, decoding, dataset):
    features, labels, recordings = [], [], []
    for i, x in enumerate(dataset):
        env.clear()
        out = env(decoding, inputs=x, encoding=encoding)
        features.append(out["features"])
        labels.append(target(x))
        recordings.append(out["times_per_channel"])
        print(f"  trial {i + 1}/{len(dataset)}", end="\r")
    print()
    return np.array(features), np.array(labels), recordings


# ===========================================================================
# 8. READOUT + EVALUATION  (ridge: still linear, just doesn't memorize noise)
# ===========================================================================
def evaluate(X, y, name=None, seeds=(0, 1, 2, 3, 4)):
    """Mean and standard deviation of test accuracy over several train/test splits."""
    accs = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        readout = RidgeClassifier(alpha=RIDGE_ALPHA).fit(X_train, y_train)
        accs.append(np.mean(readout.predict(X_test) == y_test))
    if name:
        print(f"{name}: test accuracy {np.mean(accs):.0%} ± {np.std(accs):.0%}")
    return float(np.mean(accs)), float(np.std(accs))


# ===========================================================================
# 9. MAIN  (one probe delay; sweep_probe.py runs the curve)
# ===========================================================================
def main():
    env = build_reservoir()
    encoding, decoding = make_pipeline(env)
    dataset = make_dataset()
    print(f"Running {len(dataset)} trials, probe {PROBE_DELAY_MS} ms after the conditioning zap...")
    X, y, _ = collect_trials(env, encoding, decoding, dataset)
    evaluate(X, y, f"PROBE at +{PROBE_DELAY_MS:.0f} ms: can the readout tell if a zap came before?")
    env.close()


PLAYGROUND = {
    "tagline": "A probe zap always fires. Can the readout tell whether <b>another</b> zap came 6 ms before it?",
    "input_names": ["zap before?"],
}


if __name__ == "__main__":
    main()
