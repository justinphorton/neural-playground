"""YOUR EXPERIMENT: copy this file (e.g. to my_parity.py), then work top to bottom.

This file has the exact same sections, in the same order, as demo_xor.py.
Stuck on a section? Open demo_xor.py and look at the same section there.

=============================================================== PLAN FIRST
Fill this out BEFORE writing any code. If you can't fill in a line, you're
not ready to code yet. Talk it through with the team.

TASK:        what question should the readout answer?

            IDEA: can we quantify how long the network remembers a pulse for?
            for XOR, readout seemed to vary based on timing of pulses and/or additional activation 
            of electrodes. 
                          
INPUT:       what does one example look like?

             one bit "cue" pulse, one measuring pulse. readout must answer whether there was a cue. 

ENCODING:    how does an input become electrode pulses? which electrode(s), when, how strong?
             
            the cue pulse is sent only if the bit is 1, and the probe pulse is always sent.
            probe pulse is sent at different times (6, 15, 30, 60, 120 ms) to find gap where accuracy hits 50% (random guessing). 

DECODING:    which numbers do you pull out of the spikes?
            spike counts per electrode in 5 ms bins for 30 ms after probe

HYPOTHESIS:  what do you expect, and why?
            at short gaps, network should still be "tired" from previous cue, so probe's burst looks different and
            accuracy is high. at long gaps network has recovered, and accuracy drops to 50%. 

METRIC:      how do you score it?
            test accuracy at each gap, 50% is how long the network remembers
===========================================================================
"""

import os

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# ===========================================================================
# 1. SETTINGS
# ---------------------------------------------------------------------------
# WHAT:  every knob of the experiment, in one place, so it's easy to tweak.
# FILL:  change the numbers to fit your task. The defaults are what worked
#        for XOR, so they're a sane starting point.
# ===========================================================================
N_CELLS = 300  # neurons in the reservoir (more = richer but slower)
CUE_MS = 20 #when cue pulse arrives with inpuut 1 
DELAYS_MS = [15, 20, 25, 30]  # zoom in on where accuracy drops
WINDOW_MS = 30  # how long after probe pulse to count spikes for decoding       
TRIAL_MS = int(CUE_MS + max(DELAYS_MS) + WINDOW_MS)  # how long one trial is simulated, in ms; must fit all your pulses
AMPLITUDE_UA = 40.0  # pulse strength in microamps (livn refuses above ~150)
BIN_MS = 5  # decoding time resolution: count spikes per this many ms
TRIALS_PER_INPUT = 100  # repeats per input (each repeat has different background noise)

# Optional: train on slightly varied pulses so small input changes don't break the
# readout. Off by default (0 and None). XOR uses JITTER_MS = 1.5 and (30.0, 50.0).
# With variation on, use many more trials (XOR uses 250 per input), because plain
# linear regression memorizes noise when it has few examples.
JITTER_MS = 0.0  # each training pulse lands up to this far from its nominal time
AMPLITUDE_RANGE_UA = None  # e.g. (30.0, 50.0): each training pulse's strength is drawn from this range

# livn downloads the network's wiring into ./systems/ (gitignored)
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo folder


# ===========================================================================
# 2. TASK DEFINITION
# ---------------------------------------------------------------------------
# WHAT:  the list of possible inputs, and the right answer for each one.
#        This is pure Python: no neurons involved yet.
# FILL:  INPUTS: every distinct input. Make each one a tuple of numbers, because
#                the baseline (section 7) trains directly on these.
#        target(): the correct label (0 or 1) for one input.
# EXAMPLE (XOR):  INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
#                 def target(x): return x[0] ^ x[1]
# ===========================================================================
INPUTS = [(0,), (1,)]  # TODO # 0 : no cue, 1 : cue pulse


def target(x):
    return x[0] # is there a cue pulse? 0 : no, 1 : yes


# ===========================================================================
# 3. DATA GENERATION
# ---------------------------------------------------------------------------
# WHAT:  the full list of trials to run. Same input many times, because the
#        neurons are noisy and the readout needs many examples to learn from.
# FILL:  usually nothing. Change it only if you want random inputs instead of
#        "every input, TRIALS_PER_INPUT times".
# ===========================================================================
def make_dataset():
    return [x for _ in range(TRIALS_PER_INPUT) for x in INPUTS]


# ===========================================================================
# 4. ENCODING  (input -> electrical stimulation)
# ---------------------------------------------------------------------------
# WHAT:  your "stimulator". It turns one input into a table of electrode
#        currents: one row per time step, one column per electrode (128).
#        livn then works out what each neuron feels (closer = stronger).
# FILL:  pulses(): given one input, return a list of (time in ms, electrode)
#        pairs. __call__ then turns each pair into a 1 ms pulse for you.
#        That list IS your encoding: playground.py draws it on its timeline.
# EXAMPLE (XOR): bit 1 at 20 ms, bit 2 at 26 ms, only the bits that are 1:
#            onsets = (20.0, 26.0)
#            return [(t, ch) for bit, t in zip(inputs, onsets) if bit == 1 for ch in self.channels]
# TIPS:  - Timing is everything. For the network to MIX two inputs, put them
#          a few ms apart (XOR: 6 ms worked, 30 ms failed).
#        - self.channels is a list of electrodes (set in make_pipeline). Firing
#          two neighbouring electrodes together for each input recruits more
#          neurons and made XOR noticeably more reliable. You can also pulse
#          different electrodes for different inputs.
# ===========================================================================
class CueProbeEncoding(Encoding):
    """An optional cue pulse, then a probe pulse that ALWAYS arrives delay_ms later."""

    channels: list[int]  # the electrodes each pulse fires together (two neighbours)
    amplitude_ua: float = AMPLITUDE_UA
    delay_ms: float = DELAYS_MS[0]  # gap between cue and probe
    dt: float = 0.1  # time resolution of the stimulus, in ms

    def pulses(self, inputs):
        """The input as a list of (time in ms, electrode) pulses."""
        times = [CUE_MS] if inputs[0] == 1 else []  # the cue, only if the input is 1
        times.append(CUE_MS + self.delay_ms)  # the probe, every trial
        return [(t, ch) for t in times for ch in self.channels]

    def __call__(self, env, t_end, inputs):
        # One row per time step, one column per electrode, values in microamps.
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)
        for t, ch in self.pulses(inputs):  # each pulse lasts 1 ms
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = self.amplitude_ua
        return env.cell_stimulus(electrodes, dt=self.dt)

# ===========================================================================
# 5. DECODING  (spikes -> feature vector)
# ---------------------------------------------------------------------------
# WHAT:  your "recording rig". Like a real MEA, you only see spikes per
#        ELECTRODE, not per neuron. You turn them into a list of numbers.
# FILL:  build `features` from `times_per_channel`
#        ({electrode id: array of spike times in ms}).
# RULES: - The feature vector must be the SAME LENGTH every trial.
#        - Loop over env.io.channel_ids and use .get(ch, []): silent
#          electrodes are missing from the dict.
# IDEAS: spike count per electrode (128 numbers); counts per electrode per
#        time bin (what XOR uses); time of first spike per electrode.
# ===========================================================================
class MyDecoding(Decoding):
    duration: int = TRIAL_MS  # livn quirk: the trial length lives here
    bin_ms: int = BIN_MS
    start_ms: float = 0.0

    def setup(self, env):
        env.record_spikes()  # (leave this) tell livn to record spikes

    def __call__(self, signal, env=None):
        # (leave this) group the spikes by which electrode "hears" them
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)

        # Count spikes per electrode in 5 ms bins, ONLY from the probe onward
        edges = np.arange(self.start_ms, self.duration + 1e-9, self.bin_ms)
        counts = [np.histogram(times_per_channel.get(ch, []), bins=edges)[0] for ch in env.io.channel_ids]
        features = np.concatenate(counts)  # 128 electrodes x 6 bins = 768 numbers
        return {
            "features": features,
            # raw spikes, handy if you want to plot a raster like demo_xor.py
            "spike_ids": np.asarray(signal.spike_ids),
            "spike_times": np.asarray(signal.spike_times),
        }


# ===========================================================================
# 6. RESERVOIR  (create the livn environment once, reuse it for everything)
# ---------------------------------------------------------------------------
# WHAT:  builds ONE fixed network of neurons. It is never trained, and the
#        same one is used for every trial and every input.
# FILL:  usually nothing. Change N_CELLS in settings to resize it.
#        make_pipeline() is what both main() and playground.py use to get
#        your Encoding + Decoding; edit it if your Encoding needs more settings.
# ===========================================================================
def build_reservoir():
    env = Env(predefined("EI", download_directory=DATA_DIR))  # standard simulated culture
    env.selection(N_CELLS, method="patch")  # simulate a small central patch of it
    env.init().apply_model_defaults()  # standard tuned weights + background noise
    return env


def strongest_electrodes(env, n=1):
    """The n electrodes that couple most strongly into our patch of cells."""
    reach = np.asarray(env.channel_reach()).sum(axis=1)
    return [int(ch) for ch in np.argsort(reach)[::-1][:n]]


def make_pipeline(env, delay_ms=DELAYS_MS[0]):
    """The Encoding + Decoding this task uses (playground.py calls this too).

    TODO (optional): use strongest_electrodes(env, n=2) to give the encoding two
    electrodes (XOR uses the strongest one plus its nearest neighbour).
    """
    probe_ms = CUE_MS + delay_ms
    encoding = CueProbeEncoding(channels=strongest_electrodes(env, n=2), delay_ms=delay_ms)
    decoding = MyDecoding(duration=int(probe_ms + WINDOW_MS), start_ms=probe_ms)
    return encoding, decoding


# ===========================================================================
# 7. COLLECT NEURAL FEATURES  (run every trial through the neurons)
# ---------------------------------------------------------------------------
# WHAT:  for each trial: reset, then encode -> simulate -> decode.
# FILL:  nothing. This is the same for every task.
# ===========================================================================
def collect_trials(env, encoding, decoding, dataset):
    features, labels = [], []
    for i, x in enumerate(dataset):
        # Fair trials: back to t=0 with fresh noise, same wiring
        env.clear()
        # livn's encode -> simulate -> decode in one call
        out = env(decoding, inputs=x, encoding=encoding)
        features.append(out["features"])
        labels.append(target(x))
        print(f"  trial {i + 1}/{len(dataset)}", end="\r")
    print()
    return np.array(features), np.array(labels)


# ===========================================================================
# 8. READOUT + EVALUATION  (the only part that learns)
# ---------------------------------------------------------------------------
# WHAT:  fit linear regression on 70% of trials, test on the other 30%.
#        Run it TWICE: on the raw inputs (baseline) and on the neural
#        features. The reservoir only "helped" if it beats the baseline.
# FILL:  nothing for yes/no tasks. Linear regression outputs one number per
#        trial, and output > 0.5 means "1".
# RULE:  keep the readout linear. If a fancy model solves the task, you can't
#        tell whether the neurons did the work or the model did.
# ===========================================================================
def evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    readout = Ridge(alpha=10.0).fit(X_train, y_train)  # still linear; steadier with few trials
    accuracy = np.mean((readout.predict(X_test) > 0.5) == y_test)  # > 0.5 means "1"
    print(f"{name}: test accuracy {accuracy:.0%}")
    return accuracy


# ===========================================================================
# 9. MAIN  (wire it all together)
# ---------------------------------------------------------------------------
# FILL:  nothing, unless your Encoding needs more settings (e.g. more channels).
# ===========================================================================
def main():
    env = build_reservoir()
    dataset = make_dataset()

    accuracy = {}
    for delay in DELAYS_MS:
        encoding, decoding = make_pipeline(env, delay)
        print(f"Gap {delay} ms: running {len(dataset)} trials...")
        X_reservoir, y = collect_trials(env, encoding, decoding, dataset)
        accuracy[delay] = evaluate(X_reservoir, y, f"GAP {delay} ms  readout on spikes after the probe")

    print("\nSUMMARY  can the network still tell a cue came earlier?  (50% = coin flip)")
    for delay, acc in accuracy.items():
        print(f"  gap {delay:>4} ms   accuracy {acc:.0%}")
    env.close()


# ===========================================================================
# PLAYGROUND TEXT (optional)
# ---------------------------------------------------------------------------
# WHAT:  text for the web playground (python playground/web.py <this file>).
#        The page already has generic text for every section and computes the
#        "best straight line on the raw inputs" baseline for your task by itself.
#        Anything you put here replaces one section's generic text. Leave it
#        empty and the page still works. See demo_xor.py for a full example.
# KEYS (all optional; values are HTML strings):
#        "tagline"      one line under the page title
#        "input_names"  axis labels for 2-input tasks, e.g. ["bit 1", "bit 2"]
#        "encoding"     how your inputs become pulses (step 1)
#        "howto"        a "try this by hand" tip (step 1)
#        "raster"       what to look for in the spikes (step 2)
#        "windows"      what each time window ended up meaning (step 4)
#        "baseline"     why a line on the raw inputs does / doesn't work (step 7)
#        "training"     how your readout was trained / tested (step 9)
#        "ideas"        a LIST of "things to try" (the big-picture card)
# NOTE:  the playground handles: yes/no answers (0 or 1), 1 ms pulse encodings
#        with an `amplitude_ua` setting, up to 4 electrodes on its timeline, and
#        electrode x time-window features. See "Playground scope" in the README.
# ===========================================================================
PLAYGROUND = {}


if __name__ == "__main__":
    main()
