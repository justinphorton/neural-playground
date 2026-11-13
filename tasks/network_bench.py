"""Task-free benchmarks for a simulated culture: response shape, memory and interaction.

    python tasks/network_bench.py --variant raw       # see VARIANTS below

Pick a network by these numbers, not by how well it plays a particular game, so
that a later task result isn't the product of tuning the network to that task.

1. RESPONSE     spikes per millisecond after one pulse; how much comes later than 2 ms.
2. MEMORY       which of 16 electrodes was pulsed, decoded from a 5 ms window d ms later.
3. INTERACTION  a two-pulse XOR (A, B, both or neither) with B on the same electrode,
                a neighbour or a far one, at several gaps. A straight line on the raw
                pulses can't solve XOR, so anything above 50% means the pulses interact.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from livn.env import Env
from livn.system import predefined
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(DATA_DIR, "results", "network_bench")
TRIAL_MS = 100
PULSE_MS = 20.0
WORKERS = 4

SCREEN = [72, 73, 55, 54, 76, 77, 51, 50, 124, 125, 3, 2, 120, 121, 7, 6]  # the 16 strongest electrodes
A_ELECTRODE = 77
B_ELECTRODES = {"same": 77, "neighbour": 51, "far": 6}  # 0, 200 and ~570 um from A
GAPS = [3, 6, 12, 25]
MEMORY_DELAYS = [0, 2, 5, 10, 20, 40, 60]
MEMORY_REPS = 25
XOR_REPS = 15
SPONTANEOUS_TRIALS = 20

TUNED = os.path.join(DATA_DIR, "systems", "graphs", "EI", "params", "default.json")
VARIANTS = {
    "raw": {"loader": "model", "about": "what the repo has used: apply_model_defaults(), i.e. untuned"},
    "tuned": {"loader": "tuned", "about": "livn's tuned EI parameters (fitted on the full culture)"},
    "raw_silenced": {"loader": "model", "silence": True, "about": "raw, all connections set to zero"},
    "tuned_silenced": {"loader": "tuned", "silence": True, "about": "tuned, all connections set to zero"},
    "tuned_strong_exc": {"loader": "tuned", "scale": {"EXC_EXC-dend-AMPA-weight": 2.0,
                                                      "EXC_EXC-dend-NMDA-weight": 2.0},
                         "about": "tuned, excitatory-to-excitatory connections doubled"},
    "tuned_amp80": {"loader": "tuned", "amplitude": 80.0, "about": "tuned, 80 uA pulses instead of 40"},
    "full_tuned": {"loader": "tuned", "full": True,
                   "about": "the whole 2,600-cell culture with the tuned parameters it was fitted on"},
    "full_tuned_silenced": {"loader": "tuned", "full": True, "silence": True,
                            "about": "the whole culture, tuned, all connections set to zero"},
    "E_full": {"culture": "E", "loader": "tuned", "full": True, "amplitude": 15.0, "memory_only": True,
               "about": "excitatory-only culture E, whole culture, its tuned parameters (bursting), 15 uA"},
    "E_full_silenced": {"culture": "E", "loader": "tuned", "full": True, "amplitude": 15.0, "memory_only": True,
                        "silence": True, "about": "E, whole culture, tuned, all connections set to zero"},
}


def build(variant):
    cfg = VARIANTS[variant]
    env = Env(predefined(cfg.get("culture", "EI"), download_directory=DATA_DIR))
    if not cfg.get("full"):
        env.selection(300, method="patch")
    env.init()
    if cfg["loader"] == "tuned":
        env.apply_default_params()
    else:
        env.apply_model_defaults()
    if cfg.get("scale"):
        tuned = json.load(open(TUNED))["ReducedCalciumSomaDendrite"]["default"]["params"]
        env.set_params({k: tuned[k] * s for k, s in cfg["scale"].items()})
    if cfg.get("silence"):
        env.set_params({w: 0.0 for w in env.weight_names})
    env.record_spikes()
    return env


def run_trial(env, pulses, amplitude, dt=0.1):
    """pulses: [(time_ms, electrode)], biphasic 1 ms. Returns spikes per electrode per ms."""
    current = np.zeros((int(TRIAL_MS / dt), env.io.num_channels), dtype=np.float32)
    half = int(0.5 / dt)
    for t, ch in pulses:
        s = int(t / dt)
        current[s:s + half, ch] = -amplitude
        current[s + half:s + 2 * half, ch] = amplitude
    env.clear()
    rec = env.run(TRIAL_MS, env.cell_stimulus(current, dt=dt))
    _, times = env.channel_recording(rec.spike_ids, rec.spike_times)
    edges = np.arange(TRIAL_MS + 1)
    return np.stack([np.histogram(times.get(ch, []), bins=edges)[0] for ch in env.io.channel_ids]).astype(np.uint8)


def _job(job):
    variant, stream, trials = job
    env = build(variant)
    env.reseed_noise(stream=stream)
    amp = VARIANTS[variant].get("amplitude", 40.0)
    out = [run_trial(env, pulses, amp) for pulses in trials]
    env.close()
    return out


def simulate(variant, trials):
    chunks = [trials[k::WORKERS] for k in range(WORKERS)]
    base = random.randint(10**6, 10**8)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        parts = list(pool.map(_job, [(variant, base + k * 100_000, c) for k, c in enumerate(chunks)]))
    out = np.zeros((len(trials), 128, TRIAL_MS), dtype=np.uint8)
    for k, part in enumerate(parts):
        out[k::WORKERS] = np.array(part)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="raw")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = VARIANTS[args.variant]
    screen = SCREEN
    if cfg.get("culture", "EI") != "EI":  # another culture: its own 16 strongest electrodes
        probe = build(args.variant)
        reach = np.asarray(probe.channel_reach()).sum(axis=1)
        screen = [int(c) for c in np.argsort(reach)[::-1][:16]]
        probe.close()

    # every trial of every test, as a list of pulses, with a tag saying what it is
    trials, tags = [], []
    for _ in range(SPONTANEOUS_TRIALS):
        trials.append([]); tags.append(("spont",))
    for e in screen:
        for _ in range(MEMORY_REPS):
            trials.append([(PULSE_MS, e)]); tags.append(("memory", e))
    for rel, b in ({} if cfg.get("memory_only") else B_ELECTRODES).items():
        for gap in GAPS:
            for a_on in (0, 1):
                for b_on in (0, 1):
                    for _ in range(XOR_REPS):
                        p = ([(PULSE_MS, A_ELECTRODE)] if a_on else []) + ([(PULSE_MS + gap, b)] if b_on else [])
                        trials.append(p); tags.append(("xor", rel, gap, a_on, b_on))

    started = time.time()
    counts = simulate(args.variant, trials)
    minutes = (time.time() - started) / 60
    np.savez_compressed(os.path.join(OUT_DIR, f"{args.variant}.npz"), counts=counts)

    t0 = int(PULSE_MS)
    spont = counts[[i for i, t in enumerate(tags) if t[0] == "spont"]]
    mem_idx = [i for i, t in enumerate(tags) if t[0] == "memory"]
    mem = counts[mem_idx].astype(float)
    labels = np.array([tags[i][1] for i in mem_idx])

    # 1. response shape (minus the spontaneous background per ms)
    background = spont.sum(1).mean()  / TRIAL_MS
    per_ms = mem.sum(1).mean(0) - background
    after = per_ms[t0:]
    total = after[:70].clip(0).sum()
    shape = {"spontaneous_hz_per_neuron": float(spont.sum() / len(spont) / 300 / (TRIAL_MS / 1000)),
             "evoked_spikes_per_pulse": float(total),
             "share_0_2ms": float(after[:2].clip(0).sum() / max(total, 1e-9)),
             "share_2_10ms": float(after[2:10].clip(0).sum() / max(total, 1e-9)),
             "share_10_60ms": float(after[10:70].clip(0).sum() / max(total, 1e-9))}

    # 2. memory: which electrode, from a 5 ms window d ms after the pulse
    memory = {}
    for d in MEMORY_DELAYS:
        X = mem[:, :, t0 + d:t0 + d + 5].sum(2)
        memory[d] = float(np.mean(cross_val_score(LogisticRegression(max_iter=3000, C=0.5), X, labels,
                                                   cv=StratifiedKFold(5, shuffle=True, random_state=0))))

    # 3. interaction: XOR from the 5 ms bins after the first pulse, straight-line readout.
    # Trials are stored condition by condition, so the folds must be shuffled.
    xor = {}
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for rel in ({} if cfg.get("memory_only") else B_ELECTRODES):
        for gap in GAPS:
            idx = [i for i, t in enumerate(tags) if t[0] == "xor" and t[1] == rel and t[2] == gap]
            X = counts[idx][:, :, t0:t0 + 60].reshape(len(idx), 128, 12, 5).sum(3).reshape(len(idx), -1).astype(float)
            y = np.array([tags[i][3] ^ tags[i][4] for i in idx])
            xor[f"{rel}_{gap}ms"] = float(np.mean(cross_val_score(RidgeClassifierCV(alphas=[1, 10, 100, 1000]), X, y, cv=folds)))

    res = {"variant": args.variant, "about": VARIANTS[args.variant]["about"], "minutes": minutes,
           "response": shape, "memory_by_delay_ms": memory, "xor_accuracy": xor}
    with open(os.path.join(OUT_DIR, f"{args.variant}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== {args.variant}: {VARIANTS[args.variant]['about']}  ({minutes:.1f} min)")
    print(f"  spontaneous {shape['spontaneous_hz_per_neuron']:.2f} Hz/neuron | evoked {shape['evoked_spikes_per_pulse']:.0f} spikes/pulse | "
          f"share 0-2 ms {shape['share_0_2ms']:.0%}, 2-10 ms {shape['share_2_10ms']:.0%}, 10-60 ms {shape['share_10_60ms']:.0%}")
    print("  memory (which of 16 electrodes, chance 6%): " + "  ".join(f"{d}ms {a:.0%}" for d, a in memory.items()))
    for rel in ({} if cfg.get("memory_only") else B_ELECTRODES):
        print(f"  XOR, B on {rel:9s}: " + "  ".join(f"gap {g}ms {xor[f'{rel}_{g}ms']:.0%}" for g in GAPS))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
