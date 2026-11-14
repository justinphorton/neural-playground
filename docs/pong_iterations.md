# Pong on a simulated culture: what was tried and what it showed

Faisal's branch. Each design answers three questions:

- **Runs?** Does the experiment work end to end without bugs?
- **Beats the line?** Does a straight-line readout on the neurons beat the *same* straight-line readout on the raw input, with no neurons?
- **Neurons get credit?** If it beats the line, is the gain really caused by the neurons (checked with controls)?

All runs use livn 2.0.0 and the `EI` culture, 300-neuron patch, unless stated. Scores are on held-out balls.

## Summary

| # | Design | File | Runs? | Beats the line? | Neurons get credit? |
|---|---|---|---|---|---|
| 1 | Closed-loop game: the culture plays every frame | `tasks/pong.py` | Yes | **No** (48% vs 55%) | No |
| 2 | Team spec: falling ball, predict the landing spot | `tasks/pong_spec.py` | Yes | **No** (ties; at chance with the 2 ms blind phase) | No |
| 3 | Software reservoir in place of the culture | `tasks/pong_reservoir_sweep.py` | Yes | **Yes**, by up to +21 points | Yes: needs recurrence |
| 4 | Probe pulse to read the culture's hidden memory | `tasks/pong_probe.py` | Yes | **No** | No |
| 5 | Boosted culture: connections x30 | `pong_spec.py --exc-scale 30` | Yes | **No** (ties) | No |

**Bottom line:** the game, encoding, readout and controls all work, and the team spec is a valid test: a line fails at medium difficulty while a flexible decoder succeeds. A software reservoir with strong recurrence beats the line, so the task *can* reward a network. But the livn culture never beats the line. By default it forgets a pulse within about 2 ms. With its connections boosted, it locks "on" after the ball's first pulse instead of keeping a fading memory.

**Note for the team:** every task file calls `apply_model_defaults()` ("standard tuned weights"), but for the `EI` culture that loads nothing; livn only ships built-in defaults for EI1–EI4. The tuned values load only through `apply_default_params()`, and the two give very different cultures (0.71 vs 0.02 Hz background firing per neuron). All results here use `apply_model_defaults()`, like the rest of the repo.

## 1. Closed-loop game (`tasks/pong.py`)

- **What:** every frame, three pulses on a 4 × 4 grid of electrodes (ball 4 frames ago, ball now, paddle), 6 ms apart. A line predicts how far the landing spot is from the paddle, and the paddle moves there. The culture is in the loop every frame. Before any neurons were used, the game was played without them, which caught three bugs: a paddle too fast to miss, broken wall bounces, and controllers trained to steer timidly.
- **Result (100 balls, same for every controller):**

| Controller | Hit rate |
|---|---|
| Do nothing | 32% |
| Follow the ball | 41% |
| **Neurons + line** | **48%** |
| Line on the 4 × 4 pulses, no neurons | 55% |
| Best decoder on the 4 × 4 pulses (ceiling) | 64% |
| Perfect player | 92% |

- **Runs?** Yes, the culture really plays. **Beats the line?** No: 48% vs 55% (±5 points of noise).
- **Why:** every pulse can be read back from the spikes, but the three pulses mostly hit different neurons, so they don't interact. The spikes are a noisy copy of the pulses. The coarse 4 × 4 grid also caps any decoder at 64%.

## 2. Team spec: falling ball, landing prediction (`tasks/pong_spec.py`)

- **What:** the team lead's spec. A ball falls with a varying sideways speed and bounces off the side walls. There's one biphasic pulse each time the ball enters a new grid cell; neighbouring cells use neighbouring electrodes. The paddle is never encoded. At the commit point, a line predicts the landing position. It's compared with the same line on the raw pulse history, which has perfect memory and no noise. 2,000 balls per level, tested on the later 30%. Beginner walkthrough: `tasks/pong_spec.md`.
- **Result (landing side correct, 600 test balls):**

| | Easy | Medium | Hard |
|---|---|---|---|
| Line on pulse history (baseline) | 67% | 52% | 51% |
| Best decoder on pulse history | 88% | 89% | 91% |
| Neurons + line, no blind phase | 66% | 52% | 56% |
| Neurons + line, 2 ms blind phase | 53% | 51% | 49% |
| Best decoder on the neurons | 85% | 89% | 86% |

- **Runs?** Yes. Controls behave: shuffled labels give chance, and there's no carry-over between balls.
- **Beats the line?** No. Without the blind phase the neurons tie the line (within ±4 points). With it, the readout is at chance.
- **Why:** the information is in the spikes (a flexible decoder reads 85–89%), but not in a form a line can use better than the raw pulses. Almost the whole response comes within 1 ms of each pulse, so a 2 ms blind phase deletes about 98% of the spikes. Measured separately, with no game, the culture can tell which electrode was pulsed in the first 5 ms and is at chance after that, with or without its connections. The same holds when decoding every neuron directly, so it's the culture, not the recording.
- **What did work:** the task design. At medium, the line is near chance (52%) while a flexible decoder gets 89%, so there's real room for a network to help.

## 3. Software reservoir: what would a network need? (`tasks/pong_reservoir_sweep.py`)

- **What:** the same game, grid and pulse rule as the team spec, but the pulses drive a software echo state network (300 units) instead of the livn culture. Its memory and mixing are adjustable, so we can see what a network must have for a line on it to beat the line on the raw pulse history.
- **Result (medium difficulty, 600 test balls, mean and range over 5 random networks):**

| Reservoir | Memory | Landing side correct | vs line (52%) |
|---|---|---|---|
| No recurrence (units don't connect) | gone after 2 ms | 52% | ties |
| Weak recurrence (radius 0.5) | about 8–16 ms | 54% | ties |
| Recurrence 0.9 | the whole 24+ ms window | 64% (61–67%) | +12 |
| **Recurrence 1.1** | the whole 24+ ms window | **71% (69–73%)** | **+19** |
| **Recurrence 1.3** | the whole 24+ ms window | **73% (72–74%)** | **+21** |
| Slow leaky memory, any recurrence | long | 54–57% | ties |

Easy and hard show the same pattern: 76% (line 67%) and 70% (line 51%).

- **Runs?** Yes. **Beats the line?** Yes, by up to 21 points, consistently. **Network gets credit?** Yes: the same reservoir without recurrence ties the line.
- **What it means:** this Pong *can* show a reservoir doing real work, but only if the network keeps activity that says *where* it was stimulated for the whole fall (24+ ms), mixed through strong recurrent connections. Slow, smooth fading alone doesn't help.
- **Why the livn culture failed:** it behaves like the no-recurrence reservoir: perfect memory for 2 ms, then nothing.
- **A requirement for real cultures:** to play this Pong, a culture must keep stimulus-specific activity for roughly 25 ms or more, carried by its connections. That's something to measure on a real dish (for example, with Lily's cue-then-probe task on main).

## 4. Probe pulse: reading the culture's hidden memory (`tasks/pong_probe.py`)

- **What:** neurons hit recently are still recovering and respond less to a new pulse, which is the effect XOR and Lily's `memory_task.py` use. So at the commit point, fire a probe on all 16 grid electrodes at once. Its response should be weaker wherever the ball passed recently. Same balls, grid, baseline and split as the team spec.
- **Result (landing side correct, 600 test balls):**

| | Easy | Medium | Hard |
|---|---|---|---|
| Line on pulse history (baseline) | 67% | 52% | 51% |
| Neurons before the probe | 63% | 55% | 53% |
| **Probe response only** | **54%** | **51%** | **52%** |
| Both | 64% | 54% | 54% |
| Shuffled-label control | 50% | 51% | 49% |

- **Runs?** Yes. **Beats the line?** No. **Neurons get credit?** No.
- **Why:** the probe does read the hidden memory, but only of where the ball *just* was (its column 83% of the time when the last pulse was within 6 ms). Predicting the landing spot needs the trajectory, and the baseline already has the exact last position.

## 5. Boosted culture: connections x30 (`pong_spec.py --exc-scale 30`)

- **What:** design 3 says Pong needs memory carried by connections, and in the default culture the connections do almost nothing. So the excitatory connections were turned up to 30x livn's tuned values. The setting was chosen **before** running Pong, by measuring how long activity still says which electrode was pulsed. `--silence` cuts every connection, as a control.
- **Result (landing side correct, 600 test balls):**

| | Easy | Medium | Hard |
|---|---|---|---|
| Line on pulse history (baseline) | 67% | 52% | 51% |
| Default culture + line | 66% | 52% | 56% |
| **Boosted culture + line** | **66%** | **55%** | **52%** |
| Connections cut (control) | — | 55% | — |
| *With the 2 ms blind phase:* | | | |
| Default culture + line | 53% | 51% | 49% |
| **Boosted culture + line** | **64%** | 50% | 50% |

- **Runs?** Yes. **Beats the line?** No: at medium, 55% equals the cut-connections control. **Network gets credit?** No.
- **Why:** a closer look showed the boosted culture doesn't hold a fading memory. One pulse locks the whole network "on", and it stays on. So the ball's first pulse flips it, and every later frame lands on a saturated network. Boosted versions with 1,000 neurons and with stronger inhibition behaved the same: either the network locks on, or it forgets within a few ms.
- **Partial win:** with the blind phase, the boosted culture keeps 64% on easy balls, where the default culture falls to chance. Its information travels in later, network-driven spikes that the blind phase doesn't delete. That matters for real hardware, where the blind phase is unavoidable.

## What's next

1. **Real cultures.** The key open question is whether a real dish keeps stimulus-specific activity for about 25 ms or more (design 3). If it does, the team spec is ready to show it.
2. **Find the culture's limit on this task.** Run the team spec with shorter frames (0.5–1 ms), so the whole fall fits inside the culture's short memory. If the culture beats the line there, its memory is the limit, and we have a number to ask of a real culture.
