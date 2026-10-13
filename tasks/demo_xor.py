"""Temporal XOR with a simulated neural reservoir (livn).

Pipeline:  bits -> Encoding -> electrode pulses -> livn network -> spikes
           -> Decoding -> feature vector -> linear regression -> XOR answer

The punchline: linear regression alone CAN'T solve XOR. Put a fixed, untrained
network of neurons in front of it, and the same linear regression CAN.
The only thing that learns is that linear regression at the end (the "readout").

Same 9 sections, in the same order, as template_task.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ======================================================== 1. SETTINGS
N_CELLS = 300  # size of our reservoir: a small patch of a simulated culture
TRIAL_MS = 60  # how long each trial is simulated
# Bit 1 arrives at 20 ms, bit 2 only 6 ms later, while the network is still
# reacting to bit 1. That overlap is what makes the response to (1,1) more than
# just "response to bit 1 + response to bit 2", the nonlinear mixing XOR needs.
# (With a 30 ms gap the network forgets bit 1 and accuracy drops to ~60%.)
BIT_TIMES_MS = (20.0, 26.0)
AMPLITUDE_UA = 40.0  # pulse strength for a "1" (a "0" is no pulse)
BIN_MS = 5  # decoding: count spikes in 5 ms time bins
TRIALS_PER_INPUT = 25  # repeats of each XOR input (noise makes every trial differ)

# livn downloads the network's wiring into ./systems/ (gitignored)
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo folder


# ======================================================== 2. TASK DEFINITION
INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def target(x):
    return x[0] ^ x[1]  # XOR: 1 if exactly one bit is on


# ======================================================== 3. DATA GENERATION
def make_dataset():
    return [x for _ in range(TRIALS_PER_INPUT) for x in INPUTS]


# ======================================================== 4. ENCODING
class TemporalBitsEncoding(Encoding):
    """Turns two bits into electrical pulses on one electrode, one after the other.

    This is our "input device": in a real lab it would be the stimulator
    attached to the multi-electrode array (MEA).
    """

    channel: int  # which electrode we stimulate through
    amplitude_ua: float = AMPLITUDE_UA
    dt: float = 0.1  # time resolution of the stimulus, in ms

    def pulses(self, inputs):
        """The input as a list of (time in ms, electrode) pulses.

        This is the whole encoding idea: (1, 1) is just "pulse at 20 ms and
        26 ms on electrode 3", (1, 0) is "pulse at 20 ms", (0, 0) is nothing.
        """
        return [(t, self.channel) for bit, t in zip(inputs, BIT_TIMES_MS) if bit == 1]

    def __call__(self, env, t_end, inputs):
        # One row per time step, one column per electrode, values in microamps.
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)

        for t, ch in self.pulses(inputs):  # each pulse lasts 1 ms
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = self.amplitude_ua

        # The MEA model works out how much each neuron feels from each electrode
        # (closer neurons feel more) and gives back a per-neuron stimulus.
        return env.cell_stimulus(electrodes, dt=self.dt)


# ======================================================== 5. DECODING
class BinnedChannelCounts(Decoding):
    """Turns recorded spikes into a fixed-length feature vector.

    Like a real MEA, we only see spikes per *electrode*, not per neuron.
    For each electrode we count spikes in each time bin, then flatten:
    128 electrodes x 12 bins = 1536 numbers per trial.
    """

    duration: int = TRIAL_MS  # livn quirk: the trial length lives here
    bin_ms: int = BIN_MS

    def setup(self, env):
        env.record_spikes()  # tell livn what to record before the run starts

    def __call__(self, signal, env=None):
        # Group the spikes by which electrode "hears" them.
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)

        edges = np.arange(0, self.duration + self.bin_ms, self.bin_ms)
        counts = [
            np.histogram(times_per_channel.get(ch, []), bins=edges)[0]
            for ch in env.io.channel_ids
        ]
        return {
            "features": np.concatenate(counts),
            # raw spikes kept only so we can draw a raster plot later
            "spike_ids": np.asarray(signal.spike_ids),
            "spike_times": np.asarray(signal.spike_times),
        }


# ======================================================== 6. RESERVOIR
def build_reservoir():
    """Create ONE fixed network. It is reused, unchanged, for every trial."""
    env = Env(predefined("EI", download_directory=DATA_DIR))  # standard simulated culture
    env.selection(N_CELLS, method="patch")  # simulate a small central patch of it
    env.init().apply_model_defaults()  # standard tuned weights + background noise
    return env


def strongest_electrodes(env, n=1):
    """The n electrodes that couple most strongly into our patch of cells."""
    reach = np.asarray(env.channel_reach()).sum(axis=1)
    return [int(ch) for ch in np.argsort(reach)[::-1][:n]]


def make_pipeline(env):
    """The Encoding + Decoding this task uses (playground.py uses it too)."""
    return TemporalBitsEncoding(channel=strongest_electrodes(env)[0]), BinnedChannelCounts()


# ======================================================== 7. COLLECT NEURAL FEATURES
def collect_trials(env, encoding, decoding, dataset):
    features, labels, examples = [], [], {}
    for i, x in enumerate(dataset):
        # Fair trials: reset to t=0 with fresh background noise, so no trial
        # "remembers" the previous one. The wiring itself never changes.
        env.clear()
        # livn's encode -> simulate -> decode in one call
        out = env(decoding, inputs=x, encoding=encoding)
        features.append(out["features"])
        labels.append(target(x))
        examples.setdefault(x, out)  # keep the first trial of each input for plotting
        print(f"  trial {i + 1}/{len(dataset)}", end="\r")
    print()
    return np.array(features), np.array(labels), examples


# ======================================================== 8. READOUT + EVALUATION
def evaluate(X, y, inputs, name):
    """Fit linear regression on 70% of trials, test on the other 30%."""
    X_train, X_test, y_train, y_test, _, inputs_test = train_test_split(
        X, y, inputs, test_size=0.3, stratify=y, random_state=0
    )
    readout = LinearRegression().fit(X_train, y_train)
    outputs = readout.predict(X_test)  # one number per trial; ideally 0 or 1
    accuracy = np.mean((outputs > 0.5) == y_test)  # answer "1" if output > 0.5

    print(f"\n{name}: test accuracy {accuracy:.0%}")
    for case in INPUTS:
        avg = outputs[(inputs_test == case).all(axis=1)].mean()
        print(f"  input {case}  should be {target(case)}   readout says {avg:.2f}")


def plot_rasters(examples, encoding):
    """One spike raster per XOR input: each tick is one neuron firing."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, bits in zip(axes, INPUTS):
        out = examples[bits]
        ax.set_facecolor("#fcfcfb")
        for onset, _ in encoding.pulses(bits):  # shade where a pulse was delivered
            ax.axvspan(onset, onset + 1, color="#6250d6", alpha=0.3, lw=0)
            ax.text(onset + 0.5, 1.01, "pulse", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="#52514e")
        ax.plot(out["spike_times"], out["spike_ids"], "|", color="#1baf7a", ms=3, mew=0.8)
        ax.set_title(f"input {bits}  ->  XOR = {target(bits)}", color="#0b0b0b", pad=14)
        ax.set_xlim(0, TRIAL_MS)
        ax.set_xlabel("time (ms)", color="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("neuron id", color="#52514e")
    fig.tight_layout()
    print("\nShowing a spike raster for each input (close the window to finish).")
    plt.show()


# ======================================================== 9. MAIN
def main():
    print("Building the reservoir (first run downloads the network)...")
    env = build_reservoir()
    encoding, decoding = make_pipeline(env)

    dataset = make_dataset()
    print(f"Running {len(dataset)} trials on electrode {encoding.channel}...")
    X_reservoir, y, examples = collect_trials(env, encoding, decoding, dataset)
    X_raw = np.array(dataset, dtype=float)  # the bits themselves, for the baseline

    # Baseline: linear regression straight on the bits. The best straight line
    # through XOR is flat at 0.5: it answers "I don't know" for every input.
    evaluate(X_raw, y, X_raw, "BASELINE  linear regression on raw bits")
    # The exact same linear regression, but on what the neurons did.
    evaluate(X_reservoir, y, X_raw, "RESERVOIR linear regression on spike counts")

    plot_rasters(examples, encoding)
    env.close()


# ======================================================== PLAYGROUND TEXT (optional)
# Only used by playground/web.py. The page has generic text for every section; each key
# here replaces one of them with XOR-specific text. Values are HTML; "ideas" is a list.
PLAYGROUND = {
    "tagline": "Linear regression alone <b>can't</b> solve XOR. Put a fixed, untrained network of neurons in front of it, "
               "and it <b>can</b>.",
    "input_names": ["bit 1", "bit 2"],
    "encoding": "The neurons never see bits, only zaps. For XOR: bit 1 → zap at 20 ms, bit 2 → zap at 26 ms, 0 → nothing.",
    "howto": "<b>XOR by hand:</b> Clear, then pulse electrode 3 at 20 ms = (1, 0). Add 26 ms = (1, 1). "
             "Move it to 45 ms = a pattern it never trained on.",
    "raster": "Each pulse triggers a burst that dies out within ~10 ms, back to a low <b>background</b> level (never zero: "
              "there's always some noise). Bit 2 arriving during bit 1's recovery gets a smaller burst; that's what makes XOR solvable.",
    "windows": "Bit 1's burst lands in the <b>20–25 ms</b> window, so weights there learned to push toward 1 (bit 1 alone → "
               "XOR = 1). Bit 2's burst lands in <b>25–30 ms</b>. When it's fresh, as in (0, 1), it should push toward 1; right "
               "after bit 1, as in (1, 1), many neurons are still recovering, so the burst is smaller and comes from a different "
               "mix of electrodes. The weights there learned to tell those two patterns apart: the \"tired\" pattern pushes toward "
               "0 and cancels bit 1's vote. (0, 0) has almost no spikes, so its score stays near the start value, about 0.",
    "baseline": "Linear regression is <i>w₁·bit1 + w₂·bit2 + start</i>. XOR needs bit 1 to push the answer up when bit 2 is off "
                "and down when it's on; one weight can't do both. A least-squares fit gives up entirely and says 0.5 for "
                "everything; a hand-placed line can reach 75%.",
    "ideas": [
        "<b>Compare (0, 1) with (1, 1).</b> Watch bit 2's burst in the activity chart and raster: it's smaller in (1, 1). "
        "Neurons that fired for bit 1 are still recovering when bit 2 arrives. That difference is the whole reason XOR is solvable.",
        "<b>Load (1, 1) and slide bit 2 later</b> (26 → 30 → 40 ms), with <i>Run ×10</i> at each step. The second burst grows "
        "back to full size and the answer drifts toward \"1\". The network has recovered, so the two pulses no longer interact.",
        "<b>Push bit 2 very close to bit 1</b> (about 22 ms). The two pulses blur into one burst, so the network can't tell "
        "\"two pulses\" from \"one strong pulse\".",
        "<b>Weaken a pulse</b> to 15–20 µA. Little or no burst: it's below what the neurons need to fire. A (1, 1) with a weak "
        "bit 2 tends to get read as (1, 0).",
        "<b>Strengthen a pulse</b> past 100 µA. A bigger burst that reaches more electrodes. The readout was only trained at "
        "40 µA, so its answers on patterns like this are much less reliable.",
        "<b>Move a pulse to another electrode</b> (e.g. 77). Different neurons respond first. The weights were tuned to "
        "electrode 3's response pattern, so the answer becomes close to arbitrary.",
        "<b>Shift the whole pair later</b> (bits at 40 and 46 ms). Same pattern, different time. The readout learned specific "
        "time windows, not \"two pulses 6 ms apart\", so it doesn't recognize it.",
        "<b>Send nothing</b> (Clear, then Send). Only background noise: a few random spikes, no bursts, and a score near 0.",
        "<b>For any change:</b> run the original and your version ×10 (step 8) to see if the difference beats the noise, and "
        "check step 5 for tall bars in quiet windows, which mean noise swung the answer.",
    ],
}


if __name__ == "__main__":
    main()
