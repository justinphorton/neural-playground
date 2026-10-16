"""YOUR EXPERIMENT: copy this file (e.g. to my_parity.py), then work top to bottom.

This file has the exact same sections, in the same order, as demo_xor.py.
Stuck on a section? Open demo_xor.py and look at the same section there.

=============================================================== PLAN FIRST
Fill this out BEFORE writing any code. If you can't fill in a line, you're
not ready to code yet. Talk it through with the team.

TASK:        did the player beat the dealer? (binary; hands that tie are excluded)
INPUT:       a tuple of 10 numbers: 5 player card slots then 5 dealer card slots.
             each slot is 0 (no card dealt), 2-10 (face value, J/Q/K=10), or 11 (ace).
             up to 5 cards per hand covers ~99.6% of "hit until 17" outcomes.
ENCODING:    each of the 10 card slots gets its own ~6ms-spaced time window; a card's
             value becomes that many 1ms pulses; an empty slot (0) stays silent.
DECODING:    spike counts per electrode per 5ms bin (same scheme as XOR).
HYPOTHESIS:  the raw-card baseline will specifically struggle on hands with an ace
             (ambiguous 1-or-11 value) or a near-21 total (sudden bust flip), since
             both need a nonlinear decision a straight line can't make; the reservoir
             might do better if its dynamics expose that nonlinearity.
METRIC:      test accuracy, reservoir vs. baseline - broken down by "has an ace" vs.
             not, to directly check the hypothesis instead of one aggregate number.
===========================================================================
"""

import argparse
import json
import os
import time

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split


# ===========================================================================
# 1. SETTINGS
# ---------------------------------------------------------------------------
# WHAT:  every knob of the experiment, in one place, so it's easy to tweak.
# FILL:  change the numbers to fit your task. The defaults are what worked
#        for XOR, so they're a sane starting point.
# ===========================================================================
N_CELLS = 300  # neurons in the reservoir (more = richer but slower)
AMPLITUDE_UA = 40.0  # pulse strength in microamps (livn refuses above ~150)
BIN_MS = 5  # decoding time resolution: count spikes per this many ms
NUM_TRIALS = 1000  # total random hands to simulate (no fixed INPUTS list for this task)

# Encoding experiments - all overridable from the command line (see main()).
#   "amplitude": one pulse per card slot, stronger pulse = higher card
#   "count":     card value -> (value+1)//2 pulses, PULSE_GAP_MS apart
#   "space":     card value -> that many of the strongest electrodes fire at once
ENCODING = "amplitude"
N_ELECTRODES = 1  # electrodes per pulse for "amplitude" and "count" ("space" picks its own)
START_MS = 20.0  # when the first card slot fires
SLOT_MS = {"amplitude": 6.0, "count": 12.0, "space": 6.0}  # spacing between card slots
ORDER = "sequential"  # "sequential": p1..p5 then d1..d5; "interleaved": p1 d1 p2 d2 ...
PULSE_GAP_MS = 2.0  # "count" only: gap between pulses inside one card's slot
AMP_RANGE_UA = (20.0, 80.0)  # "amplitude" only: strength for a 2 ... strength for an ace
TAIL_MS = 30.0  # quiet time after the last pulse, so the decay gets recorded

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
# NOTE:  unlike XOR, blackjack hands aren't a small fixed list - there isn't a
#        short INPUTS to enumerate. We leave INPUTS empty here and generate
#        random hands directly in make_dataset() (section 3) instead.
# ===========================================================================
INPUTS = []  # not used - see make_dataset() in section 3

CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # 2-9, J/Q/K=10, ace=11
MAX_CARDS = 5  # per hand; measured: cuts forced-stand-below-17 from ~3.9% (cap=4) to ~0.4%
EMPTY = 0  # sentinel for "no card in this slot"


def hand_value(cards):
    """Best total <=21 if possible, resolving each ace as 11 or 1."""
    cards = [c for c in cards if c != EMPTY]
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def target(x):
    """1 if the player beats the dealer, 0 otherwise. x = 4 player slots + 4 dealer slots."""
    player_total = hand_value(x[:MAX_CARDS])
    dealer_total = hand_value(x[MAX_CARDS:])
    if player_total > 21:
        return 0  # player busts - loses no matter what the dealer has
    if dealer_total > 21:
        return 1  # dealer busts, player didn't - player wins no matter their total
    return int(player_total > dealer_total)  # ties should never reach here - see section 3


# ===========================================================================
# 3. DATA GENERATION
# ---------------------------------------------------------------------------
# WHAT:  the full list of trials to run. Same input many times, because the
#        neurons are noisy and the readout needs many examples to learn from.
# FILL:  usually nothing. Change it only if you want random inputs instead of
#        "every input, TRIALS_PER_INPUT times".
# NOTE:  blackjack hands are drawn randomly (with replacement) rather than
#        taken from a small fixed INPUTS list. Ties are redrawn, never kept.
# ===========================================================================
def draw_card():
    return int(np.random.choice(CARD_VALUES))


def deal_hand():
    """Two cards, then hit while under 17, up to MAX_CARDS, with replacement."""
    cards = [draw_card(), draw_card()]
    while hand_value(cards) < 17 and len(cards) < MAX_CARDS:
        cards.append(draw_card())
    return cards + [EMPTY] * (MAX_CARDS - len(cards))


def make_dataset():
    dataset = []
    while len(dataset) < NUM_TRIALS:
        player, dealer = deal_hand(), deal_hand()
        if hand_value(player) == hand_value(dealer):
            continue  # tie - redraw, never kept
        dataset.append(tuple(player + dealer))
    return dataset


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
# NOTE:  pulses() returns (time, electrode, strength) triples instead of the
#        template's (time, electrode) pairs, because "amplitude" needs a different
#        strength per card. That means the web playground won't draw it as-is.
# ===========================================================================
def trial_ms(mode):
    """Trial length that fits all 10 card slots plus the tail, rounded up to a bin."""
    last_slot = START_MS + (2 * MAX_CARDS - 1) * SLOT_MS[mode]
    last_pulse = last_slot + (5 * PULSE_GAP_MS if mode == "count" else 0.0)  # ace = 6 pulses
    return int(np.ceil((last_pulse + TAIL_MS) / BIN_MS) * BIN_MS)


class MyEncoding(Encoding):
    channels: list[int]  # electrodes this task can stimulate (set in make_pipeline)
    mode: str = ENCODING
    amplitude_ua: float = AMPLITUDE_UA
    vary: bool = False  # training: nudge each pulse's timing/strength (see JITTER_MS, AMPLITUDE_RANGE_UA)
    dt: float = 0.1  # time resolution of the stimulus, in ms

    def pulses(self, inputs):
        """[(time_ms, electrode, strength_ua), ...] for this hand. Empty slots stay silent."""
        out = []
        for slot, card in enumerate(inputs):
            if card == EMPTY:
                continue
            if ORDER == "interleaved":  # player card k fires right before dealer card k
                slot = 2 * (slot % MAX_CARDS) + slot // MAX_CARDS
            t = START_MS + slot * SLOT_MS[self.mode]
            if self.mode == "amplitude":
                lo, hi = AMP_RANGE_UA
                ua = lo + (card - 2) / (11 - 2) * (hi - lo)
                out += [(t, ch, ua) for ch in self.channels]
            elif self.mode == "count":
                out += [(t + k * PULSE_GAP_MS, ch, self.amplitude_ua)
                        for k in range((card + 1) // 2) for ch in self.channels]
            elif self.mode == "space":
                out += [(t, ch, self.amplitude_ua) for ch in self.channels[:card]]
        return out

    def __call__(self, env, t_end, inputs):
        n_steps = int(t_end / self.dt)
        electrodes = np.zeros((n_steps, env.io.num_channels), dtype=np.float32)

        # (leave this) one 1 ms pulse for every (time, electrode) from pulses().
        # With vary=True, pulses at the same time share one random nudge in time and
        # strength, so electrodes that fire together still fire together.
        nudges = {}
        for t, ch, ua in self.pulses(inputs):
            if t not in nudges:
                shift = np.random.uniform(-JITTER_MS, JITTER_MS) if self.vary else 0.0
                strength = (np.random.uniform(*AMPLITUDE_RANGE_UA)
                            if self.vary and AMPLITUDE_RANGE_UA else None)
                nudges[t] = (shift, strength)
            shift, strength = nudges[t]
            start = int((t + shift) / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = strength if strength else ua

        # (leave this) the electrode model turns currents into per-neuron stimulus
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
    duration: int = trial_ms(ENCODING)  # livn quirk: the trial length lives here
    bin_ms: int = BIN_MS

    def setup(self, env):
        env.record_spikes()  # (leave this) tell livn to record spikes

    def __call__(self, signal, env=None):
        # (leave this) group the spikes by which electrode "hears" them
        _, times_per_channel = env.channel_recording(signal.spike_ids, signal.spike_times)

        # spike count per electrode per time bin, flattened: 128 x (duration / bin_ms)
        n_bins = self.duration // self.bin_ms
        features = np.concatenate([
            np.histogram(times_per_channel.get(ch, []), bins=n_bins, range=(0, self.duration))[0]
            for ch in env.io.channel_ids
        ]).astype(np.float32)

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


def make_pipeline(env, mode=ENCODING, n_electrodes=N_ELECTRODES):
    """The Encoding + Decoding this task uses (playground.py calls this too).

    "space" needs one electrode per point of card value, so it takes the top 11.
    """
    n = 11 if mode == "space" else n_electrodes
    encoding = MyEncoding(channels=strongest_electrodes(env, n=n), mode=mode,
                          vary=JITTER_MS > 0 or AMPLITUDE_RANGE_UA is not None)
    return encoding, MyDecoding(duration=trial_ms(mode))


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
# NOTE:  two linear readouts. Plain LinearRegression is the template's. Ridge is
#        still a straight line, just penalised for huge weights - with ~2,800
#        neural features and only ~700 training hands, plain least squares can
#        fit the training set perfectly and generalise badly. Comparing the two
#        tells us whether a failure is "no information" or "too few examples".
#        Accuracy is also split by hand type, to test the hypothesis directly.
# ===========================================================================
def hand_groups(dataset):
    """Boolean masks for the subsets the hypothesis is about."""
    has_ace = np.array([11 in x for x in dataset])
    player = np.array([hand_value(x[:MAX_CARDS]) for x in dataset])
    near_21 = (player >= 19) & (player <= 23)  # either side of the bust cliff
    return {"all": np.ones(len(dataset), bool), "has_ace": has_ace,
            "no_ace": ~has_ace, "near_21": near_21}


def evaluate(X, y, train_idx, test_idx, groups, name):
    results = {}
    for readout_name, readout in [("linear", LinearRegression()), ("ridge", Ridge(alpha=10.0))]:
        readout.fit(X[train_idx], y[train_idx])
        correct = (readout.predict(X[test_idx]) > 0.5) == y[test_idx]  # > 0.5 means "1"
        results[readout_name] = {g: float(correct[m[test_idx]].mean()) for g, m in groups.items()}
        row = "  ".join(f"{g} {acc:.0%}" for g, acc in results[readout_name].items())
        print(f"{name} [{readout_name:6s}]  {row}")
    return results


# ===========================================================================
# 9. MAIN  (wire it all together)
# ---------------------------------------------------------------------------
# FILL:  nothing, unless your Encoding needs more settings (e.g. more channels).
# ===========================================================================
# NOTE:  takes command-line options so several encodings can run at once, e.g.
#          python tasks/blackjack.py --encoding count --electrodes 2 --name count_2e
#        Every run uses the same seed, so every run sees the exact same hands.
#        Features, labels and scores are saved to results/<name>/ so we can
#        re-analyse later without re-simulating.
# ===========================================================================
def main():
    global START_MS, NUM_TRIALS, ORDER, N_CELLS
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", choices=["amplitude", "count", "space"], default=ENCODING)
    parser.add_argument("--electrodes", type=int, default=N_ELECTRODES)
    parser.add_argument("--start-ms", type=float, default=START_MS)
    parser.add_argument("--slot-ms", type=float, default=None, help="override SLOT_MS for this encoding")
    parser.add_argument("--order", choices=["sequential", "interleaved"], default=ORDER)
    parser.add_argument("--cells", type=int, default=N_CELLS)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    START_MS, NUM_TRIALS, ORDER, N_CELLS = args.start_ms, args.trials, args.order, args.cells
    if args.slot_ms is not None:
        SLOT_MS[args.encoding] = args.slot_ms
    args.slot_ms = SLOT_MS[args.encoding]
    name = args.name or f"{args.encoding}_{args.electrodes}e_start{int(args.start_ms)}"
    out_dir = os.path.join(DATA_DIR, "results", name)
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(args.seed)
    dataset = make_dataset()
    X_raw = np.array(dataset, dtype=float)  # the inputs themselves, for the baseline

    env = build_reservoir()
    encoding, decoding = make_pipeline(env, args.encoding, args.electrodes)
    print(f"[{name}] {len(dataset)} trials, {decoding.duration} ms each, electrodes {encoding.channels}")
    started = time.time()
    X_reservoir, y = collect_trials(env, encoding, decoding, dataset)
    env.close()
    minutes = (time.time() - started) / 60

    # one split, shared by baseline and reservoir, so they're scored on the same hands
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    groups = hand_groups(dataset)
    results = {
        "config": vars(args) | {"trial_ms": decoding.duration, "channels": encoding.channels,
                                "minutes": minutes},
        "baseline": evaluate(X_raw, y, train_idx, test_idx, groups, "BASELINE "),
        "reservoir": evaluate(X_reservoir, y, train_idx, test_idx, groups, "RESERVOIR"),
        "test_counts": {g: int(m[test_idx].sum()) for g, m in groups.items()},
    }
    np.savez(os.path.join(out_dir, "data.npz"), X=X_reservoir, y=y, dataset=X_raw)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{name}] done in {minutes:.1f} min -> {out_dir}")


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
