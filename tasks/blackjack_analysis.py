"""Compare the blackjack runs saved in results/ and diagnose WHY each did well or badly.

    python tasks/blackjack_analysis.py

No simulation: everything is computed from the spike features each run saved.

1. SCORES     reservoir vs. baseline accuracy, overall and per hand type.
2. CARDS      can a straight line recover each card's value from the spikes?
              If not, the encoding loses information and no readout can fix it.
3. MEMORY     can the player's FIRST card still be recovered from spikes recorded
              only from 30 ms after it fired? If not, the network forgets.
4. TOTAL      can a straight line recover the player's true hand total (aces
              resolved, a nonlinear function of the cards)?
"""

import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blackjack import BIN_MS, MAX_CARDS, hand_value  # noqa: E402

MEMORY_DELAY_MS = 30  # how long after the first card the "memory" check starts looking
RESULTS =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def r2(X, target):
    """Cross-validated R^2 of a ridge line: 1 = perfectly recoverable, <=0 = not at all."""
    return float(np.mean(cross_val_score(Ridge(alpha=10.0), X, target, cv=5, scoring="r2")))


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "results.json"))):
        with open(path) as f:
            res = json.load(f)
        data = np.load(os.path.join(os.path.dirname(path), "data.npz"))
        runs.append((os.path.basename(os.path.dirname(path)), res, data))
    if not runs:
        print("no finished runs in results/ yet")
        return

    base = runs[0][1]["baseline"]["linear"]
    print("1. SCORES  (test accuracy; baseline = line on raw cards, same 300 test hands)")
    print(f"   {'run':24s} {'readout':7s} {'all':>5s} {'ace':>5s} {'no-ace':>7s} {'near21':>7s}  {'vs base':>7s}  min")
    print(f"   {'BASELINE':24s} {'linear':7s} {base['all']:5.0%} {base['has_ace']:5.0%} "
          f"{base['no_ace']:7.0%} {base['near_21']:7.0%}")
    for name, res, _ in runs:
        for readout in ("linear", "ridge"):
            s = res["reservoir"][readout]
            print(f"   {name:24s} {readout:7s} {s['all']:5.0%} {s['has_ace']:5.0%} {s['no_ace']:7.0%} "
                  f"{s['near_21']:7.0%}  {s['all'] - base['all']:+7.0%}  {res['config']['minutes']:.0f}")

    print("\n2-4. DIAGNOSTICS  (cross-validated R^2 of a ridge line; 1 = perfect, 0 = nothing)")
    print(f"   {'run':24s} {'player cards':>13s} {'dealer cards':>13s} {'1st card, late':>15s} {'player total':>13s}")
    for name, res, data in runs:
        X, hands = data["X"], data["dataset"].astype(int)
        cfg = res["config"]
        n_bins = X.shape[1] // 128
        per_bin = X.reshape(len(X), 128, n_bins)

        card_r2 = [r2(X, hands[:, k]) for k in range(2 * MAX_CARDS)]
        # "late" = bins starting MEMORY_DELAY_MS after the player's first card fired
        late_from = int((cfg["start_ms"] + MEMORY_DELAY_MS) // BIN_MS)
        late = per_bin[:, :, late_from:].reshape(len(X), -1)
        first_card_late = r2(late, hands[:, 0])
        totals = np.array([hand_value(h[:MAX_CARDS]) for h in hands])

        print(f"   {name:24s} {np.mean(card_r2[:MAX_CARDS]):13.2f} {np.mean(card_r2[MAX_CARDS:]):13.2f} "
              f"{first_card_late:15.2f} {r2(X, totals):13.2f}")

    # reference point: the same diagnostics are trivially 1.0 on the raw cards,
    # except player total, which needs the nonlinear ace/sum rule
    hands = runs[0][2]["dataset"].astype(int)
    totals = np.array([hand_value(h[:MAX_CARDS]) for h in hands])
    print(f"   {'(raw cards, no neurons)':24s} {'1.00':>13s} {'1.00':>13s} {'-':>15s} "
          f"{r2(hands.astype(float), totals):13.2f}")


if __name__ == "__main__":
    main()
