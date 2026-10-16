"""How long does the dish remember a pulse? (memory curve)

Same 9 sections, in the same order, as template_task.py / demo_xor.py.

=============================================================== PLAN FIRST
TASK:        "was there a pulse at 20 ms?"  (1 if yes, 0 if the trial was silent)
             The twist is NOT the task, it's WHEN we're allowed to look.
INPUT:       one bit: (1,) = pulse, (0,) = no pulse
ENCODING:    a 1 = one 1 ms, 40 uA pulse at 20 ms on the two strongest electrodes
             (same as one XOR bit). A 0 = nothing at all.
DECODING:    spike count per electrode, but ONLY inside a 20 ms window that
             starts DELAY_MS after the pulse. DELAY_MS = 0 means "look right
             at the burst"; DELAY_MS = 30 means "look 30 ms after it".
HYPOTHESIS:  near 100% at delay 0 and 5 (the burst itself is in the window),
             falling to chance (~50%) by delay 15-20 ms, because XOR told us the
             burst dies within ~10 ms and nothing in the dish holds the
             information after that. If accuracy stays above chance at 20-30 ms,
             the dish has longer memory than XOR suggested (e.g. inhibitory
             rebound or slow reverberation) and that is worth knowing.
METRIC:      test accuracy of the linear readout vs DELAY_MS -> a curve.
             NOTE: the raw-input baseline is 100% here (one bit is trivially
             separable), so "reservoir beats baseline" is NOT the question.
             The question is "does the information SURVIVE inside the dish?"
===========================================================================
"""

import os

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ===========================================================================
# 1. SETTINGS
# ===========================================================================
N_CELLS = 300  # neurons in the reservoir (same patch as XOR)
TRIAL_MS = 80  # longer than XOR's 60, so a 40 ms delay still leaves a 20 ms window
PULSE_MS = 20.0  # when the pulse happens (if the input is 1)
AMPLITUDE_UA = 40.0  # pulse strength in microamps
DELAY_MS = 0.0  # how long after the pulse the decoder starts looking  <-- the knob we sweep
WINDOW_MS = 20  # the decoder looks at this many ms, starting at PULSE_MS + DELAY_MS
BIN_MS = 5  # count spikes per this many ms inside the window
TRIALS_PER_INPUT = 100  # 2 inputs x 100 = 200 trials. 128 features, so we need > 128 trials.

JITTER_MS = 0.0  # no variation: we want the cleanest possible memory measurement
AMPLITUDE_RANGE_UA = None

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo folder


# ===========================================================================
# 2. TASK DEFINITION
# ===========================================================================
INPUTS = [(0,), (1,)]  # silent trial, or one pulse


def target(x):
    return x[0]  # the answer IS the input: "was there a pulse?"


# ===========================================================================
# 3. DATA GENERATION
# ===========================================================================
def make_dataset():
    return [x for _ in range(TRIALS_PER_INPUT) for x in INPUTS]


# ===========================================================================
# 4. ENCODING  (input -> electrical stimulation)
# ===========================================================================
class OnePulseEncoding(Encoding):
    channels: list[int]  # the electrodes the pulse fires (two neighbours, like XOR)
    amplitude_ua: float = AMPLITUDE_UA
    vary: bool = False
    dt: float = 0.1

    def pulses(self, inputs):
        """A 1 = one pulse at PULSE_MS on every channel. A 0 = no pulses."""
        if inputs[0] == 1:
            return [(PULSE_MS, ch) for ch in self.channels]
        return []

    def __call__(self, env, t_end, inputs):
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)
        for t, ch in self.pulses(inputs):
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = self.amplitude_ua
        return env.cell_stimulus(electrodes, dt=self.dt)


# ===========================================================================
# 5. DECODING  (spikes -> feature vector)
# ---------------------------------------------------------------------------
# The whole experiment lives here. We throw away every spike outside
# [PULSE_MS + DELAY_MS, PULSE_MS + DELAY_MS + WINDOW_MS] and count the rest
# per electrode. 128 electrodes x (WINDOW_MS / BIN_MS) bins.
# ===========================================================================
class DelayedWindowCounts(Decoding):
    duration: int = TRIAL_MS  # livn quirk: the trial length lives here
    delay_ms: float = DELAY_MS
    window_ms: int = WINDOW_MS
    bin_ms: int = BIN_MS

    def setup(self, env):
        env.record_spikes()

    def features_from_channels(self, times_per_channel, env):
        """Counts per electrode per bin, but only inside the delayed window.

        Split out so sweep_memory.py can re-bin the SAME recorded spikes at
        many delays without re-simulating (the simulation doesn't depend on
        where we look).
        """
        start = PULSE_MS + self.delay_ms
        edges = np.arange(start, start + self.window_ms + self.bin_ms, self.bin_ms)
        counts = [
            np.histogram(times_per_channel.get(ch, []), bins=edges)[0]
            for ch in env.io.channel_ids
        ]
        return np.concatenate(counts)

    def __call__(self, signal, env=None):
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)
        return {
            "features": self.features_from_channels(times_per_channel, env),
            # keep the per-electrode spike times so the sweep can re-bin them later
            "times_per_channel": {int(ch): np.asarray(t, dtype=float) for ch, t in times_per_channel.items()},
            "spike_ids": np.asarray(signal.spike_ids),
            "spike_times": np.asarray(signal.spike_times),
        }


# ===========================================================================
# 6. RESERVOIR  (unchanged from the template, plus XOR's neighbour trick)
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


def make_pipeline(env):
    main = strongest_electrodes(env)[0]
    encoding = OnePulseEncoding(channels=[main, neighbour_electrode(env, main)])
    return encoding, DelayedWindowCounts()


# ===========================================================================
# 7. COLLECT NEURAL FEATURES  (same as the template)
# ===========================================================================
def collect_trials(env, encoding, decoding, dataset):
    features, labels, recordings = [], [], []
    for i, x in enumerate(dataset):
        env.clear()
        out = env(decoding, inputs=x, encoding=encoding)
        features.append(out["features"])
        labels.append(target(x))
        recordings.append(out["times_per_channel"])  # raw spikes, for re-binning at other delays
        print(f"  trial {i + 1}/{len(dataset)}", end="\r")
    print()
    return np.array(features), np.array(labels), recordings


# ===========================================================================
# 8. READOUT + EVALUATION  (same as the template)
# ===========================================================================
def evaluate(X, y, name, verbose=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    readout = LinearRegression().fit(X_train, y_train)
    accuracy = np.mean((readout.predict(X_test) > 0.5) == y_test)
    if verbose:
        print(f"{name}: test accuracy {accuracy:.0%}")
    return accuracy


# ===========================================================================
# 9. MAIN  (one delay only; sweep_memory.py runs the whole curve)
# ===========================================================================
def main():
    env = build_reservoir()
    encoding, decoding = make_pipeline(env)

    dataset = make_dataset()
    print(f"Running {len(dataset)} trials, decoding window starts {DELAY_MS} ms after the pulse...")
    X_reservoir, y, _ = collect_trials(env, encoding, decoding, dataset)
    X_raw = np.array(dataset, dtype=float)

    evaluate(X_raw, y, "BASELINE  linear regression on the raw bit (trivially 100%)")
    evaluate(X_reservoir, y, f"RESERVOIR linear regression on spikes {DELAY_MS:.0f} ms after the pulse")
    env.close()


# ===========================================================================
# PLAYGROUND TEXT (optional)
# ===========================================================================
PLAYGROUND = {
    "tagline": "How long does the dish <b>remember</b> a pulse? The readout only sees spikes "
               f"from a 20 ms window starting {DELAY_MS:.0f} ms after the pulse.",
    "input_names": ["pulse?"],
    "encoding": "A 1 is one 1 ms, 40 µA pulse at 20 ms on the two strongest electrodes. A 0 is a silent trial.",
    "howto": "<b>Try:</b> load (1,), then drag the pulse earlier (to 10 ms). Now the burst happens before the "
             "readout's window opens, so the readout only sees the aftermath.",
    "raster": "The burst dies out within ~10 ms. The question is whether anything in the raster after that "
              "still looks different from a silent trial.",
    "baseline": "One bit is trivially separable by a line, so the baseline is 100%. That's not the point here. "
                "The point is whether the information <i>survives</i> inside the dish.",
    "ideas": [
        "<b>Slide the pulse earlier</b> so the readout's window misses the burst. Does the score drop to ~0.5?",
        "<b>Run ×10</b> on a silent trial and on a pulse trial with the window far after the pulse. If the "
        "two spreads overlap, the dish has forgotten.",
    ],
}


if __name__ == "__main__":
    main()
