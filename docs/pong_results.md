# Reservoir Pong: results so far

*Lily Jia, November 2026. All results are from the livn simulation (`EI` culture), not a real dish.*

## Summary

- **The task works as a test.** A linear readout on the raw input gets 60% on landing side, against 100% possible. On random relabellings of the same 20 input patterns it gets 81%, so the walls are what make Pong hard for a straight line.
- **With idealized recording, the reservoir helps a little.** It scores 72% vs. 60% for the linear baselines, and beats all 30 shuffled-label controls. Accuracy rises as more pulses arrive (60% → 62% → 72%), so the rows are interacting in the network. The gain is about 2–3 more of the 20 input patterns correct.
- **There's no gain on exact landing position.** Error is 0.93 cells for the reservoir vs. 0.92 for the linear baseline, and both catch 28–29% of balls.
- **The most important finding: the default simulated culture has almost no network activity after a pulse.** 96% of spikes come within 2 ms of a pulse, from neurons the electrode fires directly. A real rig is blind for about 2 ms after each pulse, so in the realistic case the readout sees about 3 spikes per ball and scores 38%. This probably applies to the XOR and memory tasks too.
- **Turning on NMDA, strengthening synapses and balancing inhibition restores network activity, but not useful signal (sections 4–5).** The best realistic setting (300 neurons, stronger synapses, two electrodes per column) scores 58% with the 2 ms blind window: above chance, but below the 61% baseline. A 1,000-neuron network with even more activity scored 53%, within chance, probably because its activity runs on regardless of the input.

## The task

Balls fall through a box 4 columns wide, bouncing off the side walls. The network sees only the first 3 rows of the fall. After that the paddle has to commit, and the ball falls 4 more rows before landing.

- **Input:** one 1 ms, 40 µA pulse per row, at 20, 25 and 30 ms, on the electrode for the column the ball is in. Height is carried by *when* a pulse arrives; sideways position by *which* electrode fires. All three pulses fit inside the ~15–20 ms window found in `memory_task.py`.
- **Electrodes:** a straight row of 4 neighbouring electrodes, ordered left to right (76, 77, 51, 50; with pairs, 124/76, 125/77, 3/51, 2/50). The four strongest electrodes overall form a 2×2 square, so the court can't just use the top 4.
- **Labels:** a ball can produce 20 different column patterns. Each pattern is labelled with the side most of its balls land on (`pong_game.py` simulates 100,000 balls). Some patterns are nearly coin flips (62–70% of balls on one side), so even a perfect readout catches only 76% of real balls on the correct side.
- **Readout:** ridge regression, with the regularization strength chosen by cross-validation on the training set, thresholded at 0.5. Every model uses the same readout; only the features change.
- **Split:** 70/30, stratified so every pattern appears in both halves.

Difficulty 4 was chosen with `pong_game.py` (no neurons). It's where the gap between the linear baseline (56% in-game) and the best possible (76%) is largest.

## 1. Default culture (300 neurons, livn defaults, 1,000 balls)

| Readout | Test acc., no blind window | Test acc., 2 ms blind | In-game acc. (no blind) |
|---|---|---|---|
| Best possible (lookup table) | 100% | 100% | 76% |
| Linear on raw columns | 60% | 60% | 56% |
| Linear on one-hot slots | 60% | 60% | 55% |
| **Reservoir (linear on spikes)** | **72%** | **38%** | **62%** |
| Random nonlinear features (noise-free reference) | 100% | 100% | 76% |

*In-game accuracy* weights each pattern by how often real balls produce it and accounts for coin-flip patterns.

**Controls (no blind window):**
- Shuffled labels average 48% over 30 repeats (95th percentile 64%). The real 72% beats every repeat.
- On random relabellings of the 20 patterns, the one-hot baseline scores 81% and the reservoir 84%. On the Pong labelling they score 60% and 72%. So the reservoir doesn't add general memorizing power; its features happen to fit the bounce better than the raw slots do.
- With only 20 distinct patterns, a single run swings by about ±10 points. That's why every control is repeated.

**Accuracy after k pulses:**

| Pulses seen | One-hot | Reservoir |
|---|---|---|
| 1 | 60% | 60% |
| 2 | 60% | 62% |
| 3 | 60% | 72% |

**Exact landing position and paddle** (`pong_landing.py`; 4,000 test balls, half of them hit a wall after the commit point; paddle 1 cell wide, 0.5 cells per row):

| Readout | Error after 3 pulses (cells) | Catch rate | Catch, bounce after commit |
|---|---|---|---|
| Follow the ball | 1.30 | 24% | 24% |
| Linear, one-hot | 0.92 | 29% | 29% |
| Reservoir | 0.93 | 28% | 27% |
| Best possible | 0.74 | 39% | 41% |

## 2. Why the 2 ms blind window erases the signal

Spike timing after each pulse, default culture:

| Time after pulse | Share of spikes |
|---|---|
| 0–0.5 ms | 65% |
| 0.5–1 ms | 18% |
| 1–2 ms | 13% |
| 2–3 ms | 0.2% |
| 3–5 ms | 1.1% |
| 5–10 ms | 0.5% |
| 10–40 ms | 2.0% |

Almost everything is neurons fired directly by the electrode. With livn's model defaults (`apply_model_defaults`), the slow excitatory NMDA synapse has weight **0**. In a 300-neuron patch there are only 459 excitatory-to-excitatory connections, about 2 per excitatory cell, so activity has nowhere to travel. The network's "memory" in XOR and the memory task is most likely directly fired cells recovering (refractoriness and synaptic depression), not activity circulating in the network.

## 3. Making the network echo

Recorded spikes that arrive 2 ms or more after a pulse, per ball (default AMPA weight 7, NMDA 0):

| Neurons | AMPA | NMDA | Spikes ≥2 ms after pulse | Share of recorded spikes | Spikes at 40–60 ms (after the last pulse) | Seconds per ball |
|---|---|---|---|---|---|---|
| 300 | 7 | 0 | 3 | 8% | 4 | 1.4 |
| 300 | 21 | 10 | 3 | 8% | 4 | 1.5 |
| 300 | 35 | 15 | 6 | 13% | 6 | 1.5 |
| 300 | 50 | 20 | 22 | 33% | 32 | 1.4 |
| 300 | 70 | 30 | 105 | 69% | 315 | 1.4 |
| 1,000 | 7 | 0 | 9 | 17% | 17 | 5.2 |
| 1,000 | 21 | 10 | 12 | 21% | 20 | 5.4 |
| 1,000 | 35 | 15 | 79 | 59% | 223 | 5.0 |
| 1,000 | 50 | 20 | 565 | 88% | 1,868 | 5.5 |

At 300 neurons, synapse strength barely matters until it's very high, because there are so few connections. 1,000 neurons have 7× more excitatory connections (3,134). The highest settings run away: activity keeps going long after the input stops.

## 4. Pong with an echoing network (500 balls each, two electrodes per column)

| Setting | Spikes per ball (no blind / 2 ms blind) | Reservoir, no blind | Reservoir, 2 ms blind | Shuffled 95th pct. (2 ms) | One-hot baseline |
|---|---|---|---|---|---|
| 300 neurons, AMPA 50, NMDA 20 | 103 / 27 | 73% | **58%** | 52% | 61% |
| 1,000 neurons, AMPA 35, NMDA 15 | 503 / 379 | 70% | 53% | 56% | 61% |

- Stronger synapses give the realistic recording something to read: 58% is above chance, but still below the linear baseline.
- More activity didn't help. The 1,000-neuron network has 14× more spikes surviving the blind window, but they carry less information about the ball. It's likely in the runaway regime, where activity is driven by the network itself rather than the input.

## 5. Balancing inhibition and adding a 4th row (300 balls each)

*November 16.* All runs: 300 neurons, two electrodes per column, recorded once and scored with and without the 2 ms blind window. Inhibitory cells receive only 131 excitatory synapses in the patch (`INH_EXC-soma-AMPA-weight`, default 7), so inhibition can't keep up when excitation grows. These runs strengthen that input 4×.

| Setting | Spikes per ball (no blind / 2 ms) | Reservoir, no blind | Reservoir, 2 ms blind | Shuffled 95th pct. (2 ms) | One-hot baseline | In-game, 2 ms (ceiling) |
|---|---|---|---|---|---|---|
| AMPA 50, NMDA 20 (reference) | 103 / 28 | 72% | 58% | 56% | 66% | 53% (76%) |
| + inhibitory input ×4 | 105 / 28 | 72% | 59% | 53% | 66% | 54% (76%) |
| AMPA 70, NMDA 30, + inhibitory input ×4 | 208 / 127 | 70% | 58% | 54% | 66% | 54% (76%) |
| 4 rows (4 ms apart), AMPA 50, NMDA 20, + inhibitory input ×4 | 138 / 39 | 74% | 49% | 55% | 72% | 50% (81%) |

- **Balancing inhibition stops the runaway but doesn't add information.** The strong network now keeps 127 spikes per ball through the blind window, 4× more than the reference, yet accuracy stays at 58%.
- **With the blind window, no setting beats the linear baseline.** All of them sit at or just above chance.
- **A 4th row hurt.** It raises the best possible in-game score from 76% to 81% and the baseline to 72%, but with the blind window the reservoir falls to chance. Caveat: 38 patterns × 8 repeats leaves few training trials per pattern.
- The one-hot baseline reads 60% at 1,000 balls but 66% at 300 balls (same task). That's the 20-pattern noise again: compare settings within a table, not across tables.

**What this says:** extra activity after the blind window is mostly the network's own dynamics, not information about where the ball was. The ball's information is carried by the first ~2 ms of direct response, which a real rig can't record. Getting past this probably needs a different kind of input, not a stronger network. For example: stimulate electrodes *away from* the recording ones, so the readout only sees activity that travelled through the network; or encode position in pulse timing or strength rather than which electrode fires.

## Caveats

- Everything here is simulation. Real cultures do produce network responses lasting tens to hundreds of ms, so the default livn culture probably understates what a dish would do.
- With labels fixed per pattern, there are effectively only 20 test cases. Differences under about 10 points from a single run are not reliable.
- The screens in section 4 used 500 balls, half the size of section 1.
- The noise-free random-features reference always reaches the best possible score. A fair reference would add noise matched to the reservoir.

## Next steps

1. Find the edge between dying out and running away: sweep NMDA alone at 1,000 neurons, or strengthen the excitatory input to inhibitory cells (default weight 7) so inhibition can stop runaway activity.
2. Check whether XOR and the memory task survive a 2 ms blind window. If they don't, that matters for moving any task to real hardware.
3. Read the extracellular voltage (`env.potential_recording`) with threshold spike detection, in place of exact spike times.
4. Try the tuned parameters in `systems/graphs/EI/params/default.json` (`apply_default_params`). A first look gave almost no response at 40 µA, so they would need stronger pulses.

## Reproduce

```sh
python tasks/pong_game.py                 # difficulty sweep, seconds, no neurons
python tasks/pong_task.py                 # section 1 (defaults), ~25 min; saves pong_trials.npz
python tasks/pong_landing.py              # landing + paddle + replay page, seconds
python tasks/pong_screen.py s300 300 '{"EXC_EXC-dend-AMPA-weight": 50, "EXC_EXC-dend-NMDA-weight": 20}' 2 25     # section 4, ~6 min
python tasks/pong_screen.py n1000 1000 '{"EXC_EXC-dend-AMPA-weight": 35, "EXC_EXC-dend-NMDA-weight": 15}' 2 25   # ~21 min
python tasks/pong_screen.py ei4 300 '{"EXC_EXC-dend-AMPA-weight": 50, "EXC_EXC-dend-NMDA-weight": 20, "INH_EXC-soma-AMPA-weight": 28}' 2 15   # section 5, ~3.5 min
python tasks/pong_screen.py rows4 300 '{"EXC_EXC-dend-AMPA-weight": 50, "EXC_EXC-dend-NMDA-weight": 20, "INH_EXC-soma-AMPA-weight": 28}' 2 8 4 4 4   # 4 rows
```

Settings for section 4 live at the top of `pong_task.py`: `N_CELLS`, `WEIGHTS`, `ELECTRODES_PER_COLUMN`, `BLANK_MS`.

| File | What it is |
|---|---|
| `tasks/pong_game.py` | The game alone: geometry, baselines and the best possible score at each difficulty |
| `tasks/pong_task.py` | Reservoir task (landing side), in the template's nine-section format |
| `tasks/pong_landing.py` | Exact landing position, paddle simulation, writes the replay page |
| `tasks/pong_screen.py` | Records one configuration once and scores it with and without the 2 ms blind window |
| `playground/pong_replay.html` | Replay page template (`pong_landing.py` fills it in) |
