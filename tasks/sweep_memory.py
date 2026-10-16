"""Memory curve: accuracy of "was there a pulse?" vs how long after the pulse we look.

    python tasks/sweep_memory.py            # full run: 200 trials, ~4 min, then a plot
    python tasks/sweep_memory.py --quick    # 40 trials, for checking that it runs

Simulates each trial ONCE and keeps the recorded spikes, then re-bins the same
spikes at every delay. The neurons don't know where the readout will look, so
re-simulating per delay would be wasted work.

Writes results/memory_curve.csv and results/memory_curve.png.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_curve as task  # noqa: E402

DELAYS_MS = [0, 5, 10, 15, 20, 25, 30, 40]
SEEDS = [0, 1, 2]  # repeat the train/test split a few times and average, so one lucky split doesn't mislead
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def features_at_delay(recordings, env, delay_ms):
    """Re-bin every trial's stored spikes with the decoder's window at `delay_ms`."""
    decoder = task.DelayedWindowCounts(delay_ms=float(delay_ms))
    return np.array([decoder.features_from_channels(r, env) for r in recordings])


def accuracy_at_delay(recordings, y, env, delay_ms):
    X = features_at_delay(recordings, env, delay_ms)
    scores = []
    for seed in SEEDS:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        readout = LinearRegression().fit(X_train, y_train)
        scores.append(np.mean((readout.predict(X_test) > 0.5) == y_test))
    # also: how many spikes land in the window on average, pulse vs silent, as a sanity check
    total = X.sum(axis=1)
    return float(np.mean(scores)), float(total[y == 1].mean()), float(total[y == 0].mean())


def main(env=None):
    quick = "--quick" in sys.argv
    if quick:
        task.TRIALS_PER_INPUT = 20

    own_env = env is None
    if own_env:
        env = task.build_reservoir()
    encoding, decoding = task.make_pipeline(env)

    dataset = task.make_dataset()
    print(f"Simulating {len(dataset)} trials once (pulse on electrodes {encoding.channels})...")
    _, y, recordings = task.collect_trials(env, encoding, decoding, dataset)

    print(f"\n{'delay':>6}  {'accuracy':>8}  {'spikes/win pulse':>16}  {'spikes/win silent':>17}")
    rows = []
    for d in DELAYS_MS:
        acc, sp, ss = accuracy_at_delay(recordings, y, env, d)
        rows.append((d, acc, sp, ss))
        print(f"{d:>5} ms  {acc:>8.0%}  {sp:>16.1f}  {ss:>17.1f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "memory_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delay_ms", "accuracy", "spikes_in_window_pulse", "spikes_in_window_silent"])
        w.writerows(rows)

    plot(rows)
    if own_env:
        env.close()
    return rows


def plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(d, [r[1] for r in rows], "o-", color="#159a6a")
    ax1.axhline(0.5, color="#999", ls="--", lw=1, label="chance")
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlabel("delay after pulse (ms)")
    ax1.set_ylabel("test accuracy: pulse vs silent")
    ax1.set_title("How long does the dish remember?")
    ax1.legend()
    ax2.plot(d, [r[2] for r in rows], "o-", color="#6250d6", label="pulse trials")
    ax2.plot(d, [r[3] for r in rows], "o-", color="#999", label="silent trials")
    ax2.set_xlabel("delay after pulse (ms)")
    ax2.set_ylabel("spikes in the 20 ms window (mean)")
    ax2.set_title("What the readout can see")
    ax2.legend()
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "memory_curve.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
