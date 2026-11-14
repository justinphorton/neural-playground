# Pong on a simulated culture: every iteration, and what it showed

Faisal's branch. Each entry answers three questions:

- **Runs?** Does the experiment work end to end without bugs?
- **Beats the line?** Does the readout on the neurons beat the *same* straight-line readout on the raw input, with no neurons?
- **Neurons get credit?** If it beats the line, is the gain really caused by the neurons (checked with controls)?

All runs use livn 2.0.0 and the `EI` culture, 300-neuron patch, unless stated. Scores are on held-out balls or trials.

## Summary

| # | Iteration | Runs? | Beats the line? | Neurons get credit? |
|---|---|---|---|---|
| 1 | "Can I still reach it?" (design check, no neurons) | Yes | n/a | n/a |
| 2 | Frame-by-frame game, first try | Bug: too easy | — | — |
| 3 | Frame-by-frame game, second try | Bug: broken bounces | — | — |
| 4 | Frame-by-frame game, third try | Bug: unfair training | — | — |
| 5 | Frame-by-frame game, final (`tasks/pong.py`) | Yes | **No** (48% vs 55%) | No |
| 6 | Team spec, 2 ms blind phase (`tasks/pong_spec.py`) | Yes | **No** (at chance) | No |
| 7 | Team spec, no blind phase | Yes | **No** (ties the line) | No |
| 8 | Network benchmark (`tasks/network_bench.py`) | Yes | n/a: measures the network | Explains 5–7 |
| 9 | Neuron-level memory check | Yes | n/a | Confirms 8 |
| 10 | Software reservoir sweep (`tasks/pong_reservoir_sweep.py`) | Yes | **Yes**, by up to +21 points | Yes: needs recurrence (control: no recurrence ties the line) |
| 11 | Probe pulse reads the culture's hidden memory (`tasks/pong_probe.py`) | Yes | **No** | No: the probe only recovers the last position |
| 12 | Boosted culture: connections x30 (`tasks/recurrence_sweep.py`, `pong_spec.py --exc-scale 30`) | Yes | **No** (ties) | No; but it survives the blind phase on easy balls |
| 13 | Bigger boosted culture: 1,000 neurons, x30 (`--cells 1000`) | Yes | **No** (53% vs 52%) | No: cut-connections control also 53% |
| 14 | Far-apart XOR on the boosted culture (`tasks/xor_far.py`) | Yes | **Yes**, 98–100% vs 75% | Yes, but it's a latch: one pulse locks the network "on" for good |
| 15 | Looking for the edge below the latch, x12–x18 (`recurrence_sweep.py --trial-ms 300`) | Yes | n/a: no setting passed, Pong not run | — |
| 16 | Boosting inhibition too, to stop the latch | Yes | n/a: no setting passed, Pong not run | — |

**Bottom line:** the game, encoding, readout and controls all work, and the team spec's task is a valid test (a line fails while a flexible decoder succeeds). But no version beats the line, because this simulated culture forgets a pulse within about 2 ms and its only interaction comes from single cells, not the network.

## 1. "Can I still reach it?" (design check, no neurons)

- **What:** given two snapshots of the ball and the paddle, predict whether the paddle can still reach the ball in time.
- **Result:** a straight line scores 64–70%, no better than always guessing "no"; a flexible model gets 83–93%.
- **Verdict:** a fair question (the inputs aren't rigged), but never run with neurons. Caveats: the inputs were exact, not the coarse zones neurons would get; the answers are imbalanced (two-thirds "no"); and the answer needs a division (distance ÷ speed), which the network has no known way to do.

## 2–4. Frame-by-frame game: three bugs caught before the real run

All three were found by playing the game *without neurons* first:

- **Too easy:** "follow the ball" scored 100% because the paddle was as fast as the ball. Fixed by making the paddle slower than the ball can climb, so you must anticipate.
- **Broken bounces:** the ball didn't reverse direction at the walls, so the perfect player scored *below* following. Fixed and verified.
- **Unfair training:** trained controllers imitated exact commands and steered timidly. Fixed: every controller steers at full speed toward its predicted landing spot.

## 5. Frame-by-frame game, final (`tasks/pong.py`)

- **What:** every frame, three pulses on a 4 × 4 grid of electrodes (ball 4 frames ago, ball now, paddle), 6 ms apart. A line predicts how far the landing spot is from the paddle; the paddle moves there. The culture is in the loop every frame.
- **Result (100 balls, same for every controller):**

| Controller | Hit rate |
|---|---|
| Do nothing | 32% |
| Follow the ball, zones only | 41% |
| **Neurons + line** | **48%** |
| Line on the 4 × 4 pulses, no neurons | 55% |
| Best decoder on the 4 × 4 pulses (ceiling) | 64% |
| Perfect player | 92% |

- **Runs?** Yes: the culture really plays.
- **Beats the line?** No. 48% vs 55% (±5 points of noise).
- **Why:** every pulse can be read back from the spikes (89–100%), but the three pulses mostly hit different neurons, so they don't interact. The spikes are a noisy copy of the pulses. The coarse 4 × 4 grid also caps any decoder at 64%.

## 6–7. Team spec: falling ball, landing prediction (`tasks/pong_spec.py`)

- **What:** the team lead's spec. A ball falls with a varying sideways speed and bounces off the side walls. One biphasic pulse each time it enters a new grid cell, neighbouring cells on neighbouring electrodes. The paddle is never encoded. At the commit point, a line predicts the landing position. Compared with the same line on the raw pulse history (perfect memory, no noise). 2,000 balls per level, tested on the later 30%. See `tasks/pong_spec.md`.
- **Result (landing side correct, 600 test balls):**

| | Easy | Medium | Hard |
|---|---|---|---|
| Line on pulse history (baseline) | 67% | 52% | 51% |
| Best decoder on pulse history | 88% | 89% | 91% |
| Neurons + line, no blind phase | 66% | 52% | 56% |
| Neurons + line, 2 ms blind phase | 53% | 51% | 49% |
| Best decoder on the neurons | 85% | 89% | 86% |

- **Runs?** Yes. Controls behave: shuffled labels at chance, no carry-over between balls.
- **Beats the line?** No. Without the blind phase the neurons tie the line (all gaps within the ±4-point noise). With it, the readout is at chance.
- **Why:** the information is in the spikes (a flexible decoder reads 85–89%), but not in a form a line can use better than the raw pulses. And almost the whole response comes within 1 ms of each pulse, so a 2 ms blind phase deletes about 98% of the spikes.
- **What did work:** the task design. At medium, the line is near chance (52%) while a flexible decoder gets 89%, so there is real room for neurons to help.

## 8. Network benchmark (`tasks/network_bench.py`)

Measures the network directly, with no game, so a network can be chosen on its properties rather than on a task score:

| Version | Remembers which electrode after 2 ms? | Two-pulse XOR (same / neighbouring electrode) | Far electrode |
|---|---|---|---|
| EI raw (what every task in the repo uses) | No | **100% / 87–90%** at 3–6 ms | chance |
| EI raw, all connections cut | No | **98% / 73–85%** | chance |
| EI tuned (patch or full culture) | No | weak (65–75% at best) | chance |
| EI tuned, stronger pulses or connections | No | no better | chance |
| E (excitatory-only, bursting), full culture | No | not run | — |

- **Memory is about 2 ms in every version.** Which electrode was pulsed is readable in the first 5 ms window, then at chance.
- **The XOR-style interaction comes from single cells:** it survives with every connection cut. It only happens when two pulses hit the same or neighbouring electrodes within about 3–12 ms. This is why XOR and rock-paper-scissors work and blackjack and Pong don't.
- **The bursting culture's late firing is background:** 32.7 spikes after a pulse vs 32.3 with no pulse, in the same window.
- **Agrees with Lily's `memory_task.py` on main:** her cue-then-probe test reads exactly this hidden "tiredness" on the same electrodes.

**Important for the whole team:** every task file calls `apply_model_defaults()` with the comment "standard tuned weights." For the `EI` culture that loads nothing; livn only has built-in defaults for the old EI1–EI4 cultures. The tuned values load only through `apply_default_params()`. The two give very different cultures (0.71 vs 0.02 Hz spontaneous firing per neuron). So every result in the repo, including the XOR demo, uses livn's untuned settings. The tuned 300-cell patch barely responds to stimulation (about 1 spike per pulse), likely because its parameters were fitted on the full culture.

## 9. Neuron-level memory check

To rule out our recording (spikes counted per electrode) as the cause, memory was decoded from every neuron's spikes directly:

| | 0–5 ms | 2–7 ms | 5–10 ms | 10–20 ms | 20–60 ms |
|---|---|---|---|---|---|
| Which electrode, from all 300 neurons | 100% | 10% | 9% | 7% | 8% |
| Spikes in the whole network in the window | 86 | 1 | 1 | 1 | 8 |
| Same with all connections cut | 100% | 7% | 7% | 7% | 6% |

The network goes silent about 2 ms after a pulse. It's the simulated culture, not the code or the recording. The memory that does exist is *hidden*: neurons still recovering from a pulse fire less when pulsed again, so only a second pulse on the same neurons can read it.

## 10. Software reservoir sweep: what would a network need? (`tasks/pong_reservoir_sweep.py`)

- **What:** the same falling-ball game, grid and pulse rule as the team spec, but the pulses drive a software echo state network (300 units) instead of the livn culture. Its memory and mixing are adjustable, so we can see what a network must have for a line on it to beat the line on the raw pulse history. Memory is measured the same way as for the culture: pulse one of 16 cells, then decode which one, some milliseconds later (2 ms per game frame).
- **Result (medium difficulty, 600 test balls, mean and range over 5 random networks):**

| Reservoir | Memory | Landing side correct | vs line on history (52%) |
|---|---|---|---|
| No recurrence (units don't connect) | gone after 2 ms | 52% | ties |
| Weak recurrence (radius 0.5) | about 8–16 ms | 54% | ties |
| Recurrence 0.9 | the whole 24+ ms window | 64% (61–67%) | +12 |
| **Recurrence 1.1** | the whole 24+ ms window | **71% (69–73%)** | **+19** |
| **Recurrence 1.3** | the whole 24+ ms window | **73% (72–74%)** | **+21** |
| Slow leaky memory (20–40 ms), any recurrence | long | 54–57% | ties |

Easy and hard levels show the same pattern: the best reservoir gets 76% (easy, line 67%) and 70% (hard, line 51%).

- **Runs?** Yes. **Beats the line?** Yes, by up to 21 points, consistently across random networks. **Network gets credit?** Yes: the same reservoir without recurrence ties the line.
- **What it means:** the team spec's Pong *can* show a reservoir doing real work, but only if the network keeps stimulus-specific activity for the whole fall (24+ ms here) and mixes it through strong recurrent connections. Slow, smooth fading alone doesn't help; it blurs the trajectory.
- **Why the livn culture failed:** it behaves exactly like the no-recurrence reservoir: perfect memory for 2 ms, then nothing, and its connections barely matter (section 8).
- **A concrete requirement for real cultures:** to play this Pong, a culture must keep activity that still says *where* it was stimulated for roughly 25 ms or more, carried by its connections. That's a measurable property to ask Archean about, or to test with Lily's cue-then-probe task on a real dish.

## 11. Probe pulse: reading the culture's hidden memory (`tasks/pong_probe.py`)

- **What:** the culture's spikes forget a pulse within 2 ms, but neurons hit recently are still recovering and respond less to a new pulse (the effect XOR and Lily's `memory_task.py` use). So at the commit point, fire a probe on all 16 grid electrodes at once. Its response should be weaker wherever the ball passed recently, turning the hidden memory back into spikes. Same balls, grid, baseline and split as the team spec.
- **Result (landing side correct, 600 test balls):**

| | Easy | Medium | Hard |
|---|---|---|---|
| Line on pulse history (baseline) | 67% | 52% | 51% |
| Neurons before the probe | 63% | 55% | 53% |
| **Probe response only** | **54%** | **51%** | **52%** |
| Both | 64% | 54% | 54% |
| Shuffled-label control | 50% | 51% | 49% |

- **Runs?** Yes; the probe triggers about 60 spikes per ball. **Beats the line?** No. **Neurons get credit?** No.
- **Why:** the probe does read the hidden memory, but only about where the ball *just* was. When the ball's last pulse came at most 6 ms before the probe, the probe response tells its column 83% of the time (chance 34%); when the last pulse was earlier, only 51% (chance 30%). Predicting the landing spot needs the ball's *trajectory* (direction and speed over several positions), and the baseline already has the exact last position. A memory of one recent position, lasting about 6 ms, adds nothing a line on the pulses doesn't already know.
- **Lesson:** hidden single-cell memory can be read out on demand, but it is too short and too local to hold a trajectory. That matches section 10: what's needed is memory carried through connections.

## 12. A boosted culture: turning the connections up (`tasks/recurrence_sweep.py`)

- **What:** section 10 showed Pong needs memory carried by connections, and in the default culture the connections do almost nothing. So the excitatory connection strengths were turned up, and each setting was measured with no game: background firing, extra spikes after a pulse, and whether activity 5–40 ms later still says which of 16 electrodes was pulsed. The setting was chosen **before** running Pong, by this rule: the best memory held across 10–40 ms with normal background firing.

| Excitatory connections (x livn's tuned values) | Background firing | Memory at 5 / 10 / 20 / 40 ms (chance 6%) |
|---|---|---|
| raw, and x1 to x10 | normal | chance |
| x20 | normal | 8–18% |
| **x30 (chosen)** | **normal, 0.18 Hz** (in 80 ms trials; see section 14) | **32% / 31% / 28% / 28%** |
| x50 | 5x higher | 56% → 18% (fades) |
| x100 | 15x higher | agitated network |

- At x30 the culture finally behaves like a recurrent network: a pulse triggers about 80 extra spikes that spread through the network, and they still say where they started 40 ms later. It's the first stimulus-specific memory beyond 2 ms seen in livn. **Caveat:** x30 is far above livn's tuned values, so this is a deliberately boosted culture, not a realistic default.
- **Pong result (landing side correct, 600 test balls):**

| | Easy | Medium | Hard |
|---|---|---|---|
| Line on pulse history (baseline) | 67% | 52% | 51% |
| Raw culture + line | 66% | 52% | 56% |
| **Boosted culture + line** | **66%** | **55%** | **52%** |
| Connections cut (control) | — | 55% | — |
| *With the spec's 2 ms blind phase:* | | | |
| Raw culture + line | 53% | 51% | 49% |
| **Boosted culture + line** | **64%** | 50% | 50% |

- **Runs?** Yes. **Beats the line?** No: at medium the boosted culture's 55% equals the cut-connections control and is within noise of the line. **Network gets credit?** No.
- **Partial win:** with the 2 ms blind phase, the raw culture falls to chance, but the boosted culture keeps 64% on easy balls, nearly matching the line. Its information travels in later, network-driven spikes that the blind phase doesn't delete. That matters for real hardware, where the blind phase is unavoidable.
- **Why it isn't enough:** the boosted culture's memory is faint (it names the right electrode about 30% of the time). The software reservoirs that beat the line held close to 100% over the whole window. Network memory exists now, but it's too weak to carry a trajectory.

## 13. A bigger boosted culture: 1,000 neurons (`--cells 1000`)

- **What:** section 12's memory was faint, so the same sweep was repeated with 1,000 neurons instead of 300. Same rule, set before Pong: the best memory held across 10–40 ms with normal background firing.

| Excitatory connections, 1,000 neurons | Background firing | Memory at 5 / 10 / 20 / 40 ms (chance 6%) |
|---|---|---|
| x20 | normal | 51% / 35% / 38% / 27% |
| **x30 (chosen)** | **normal, 0.19 Hz** (in 80 ms trials; see section 14) | **84% / 64% / 47% / 16%** |
| x50 | 3.4 Hz (runaway) | not usable |

- The memory is much clearer than at 300 neurons (84% vs 32% at 5 ms), but it **fades** by 40 ms instead of holding. The ball's trajectory spans about 48 ms, so the early frames are mostly gone by the time the readout looks.
- **Pong result, medium difficulty (600 test balls):**

| | Landing side correct |
|---|---|
| Line on pulse history (baseline) | 52% |
| **Boosted 1,000-neuron culture + line** | **53%** |
| Connections cut (control), + line | 53% |
| Shuffled labels (control) | 47% |
| *With the spec's 2 ms blind phase:* | |
| Boosted culture + line | 51% |
| Boosted culture + best decoder | 53% |
| Connections cut + best decoder | 50% |

- **Runs?** Yes (about 39 minutes per run). **Beats the line?** No. **Network gets credit?** No: the network adds nothing over the cut-connections control.
- **A trap worth naming:** with no blind phase, the best decoder (gradient boosting) reads 81% from the boosted culture, but **87% from the cut-connections culture**. That decoder is just reading each pulse's direct echo and doing the maths itself. Disconnected neurons do it better, so the credit belongs to the decoder, not the network. This is why every result here is checked against the cut-connections control.
- **Why it still isn't enough:** a clear memory that fades over about 20–40 ms isn't what Pong needs. The software reservoirs that won (section 10) held the whole window *and* mixed the frames together. The boosted culture holds part of it, and there's no sign it mixes.
- Easy and hard weren't run: the pre-set rule was to run them only if medium showed a gain.

## 14. Does the boosted network compute? Far-apart XOR (`tasks/xor_far.py`)

- **What:** a check with no game in it. Electrode A (top-left of the Pong grid) may be pulsed; electrode B (bottom-right, far away) may be pulsed 5–40 ms later. A straight line reads the spikes in the 10 ms after B and must answer **A XOR B**. A straight line on the inputs themselves can get at most 75%. Single neurons can't help, because A and B reach different neurons (section 8). So beating 75% *and* beating the cut-connections culture means the network combined A and B.

| XOR correct (400 trials per delay) | 5 ms | 10 ms | 20 ms | 40 ms |
|---|---|---|---|---|
| **Boosted x30, 1,000 neurons** | **89%** | **99%** | **98%** | **100%** |
| Connections cut, 1,000 neurons (control) | 47% | 51% | 49% | 47% |
| Boosted x30, 300 neurons | 75% | 76% | 79% | 74% |

- **Runs?** Yes. **Beats the line?** Yes: 98–100% against the 75% straight-line limit. **Network gets credit?** Yes: the cut-connections culture is at chance, and the total spike count alone gets only 46–65%, so the readout uses *which* neurons fire, not just how many. The 300-neuron culture sits at the 75% line, so it adds nothing beyond a sum.
- **But here's how it does it, and why that's a problem.** I looked at the activity over 400 ms:

| Excitatory connections | One pulse, then... | No pulse at all |
|---|---|---|
| x30, 1,000 neurons | network locks **on** (about 500 spikes per 25 ms, vs 25 at rest) and never stops | ignites by itself within about 300 ms (5 of 6 trials) |
| x20, 1,000 neurons | locks on and never stops | stays quiet |
| x30, 300 neurons | locks on and never stops | stays quiet |

- So the "memory" in sections 12–13 is mostly a **latch**: a 1-bit switch that says "something was pulsed". The XOR works because B lands differently on a network that's already on. That's real nonlinear network behaviour, but it's a degenerate regime. It can't be reset, so it can't follow a ball: the ball's first pulse flips the switch and every later frame lands on a saturated network. That's why sections 12–13 didn't help Pong.
- **A mistake of mine this caught:** my "normal background firing" rule for picking a setting looked only at 80 ms trials, which was too short to see the network ignite. The sweep now also measures latching over 300 ms trials (`recurrence_sweep.py --trial-ms 300`).
- **What it points to:** a useful reservoir must sit *just below* the latch: activity that echoes for tens of ms and then dies out. That's the regime where the software reservoirs won (section 10, recurrence about 1.1). Between x10 (no memory) and x20 (latch) there may be such a setting; that's the next run.

## 15. Looking for the edge just below the latch

- **What:** section 14 showed x20 and up locks on, and section 12 showed x10 and below has no memory. A good reservoir would sit in between: activity that echoes for tens of ms, then returns to rest. Excitation x12–x18 at 1,000 neurons, 300 ms trials. Rule set before running: pass only with clear memory at 10–40 ms **and** returning to rest.

| Excitatory connections, 1,000 neurons | Latched after one pulse | Memory at 5 / 10 / 20 / 40 ms (chance 6%) |
|---|---|---|
| x12 | 0% | 12% / 10% / 9% / 9% |
| x14 | 8% | 19% / 12% / 9% / 10% |
| x16 | **91%** | 32% / 11% / 11% / 15% |
| x18 | **100%** | 39% / 21% / 21% / 24% |

- None of them ignited without a stimulus.
- **Runs?** Yes. **Passes?** No. The change is a cliff, not a slope: between x14 and x16 the network jumps from "forgets in a few ms" to "locks on for good". There is no setting in between with lasting, fading memory. So Pong wasn't run on any of them.
- **Why, most likely:** only the excitatory connections were turned up. Once a burst starts, nothing stops it. In real cortex, inhibitory neurons stop bursts. The next run boosts inhibition along with excitation (in progress).

## 16. Boosting inhibition too, to stop the latch

- **What:** section 15's guess was that nothing stops a burst once it starts. So inhibition was boosted along with excitation. 1,000 neurons, 300 ms trials, same pass rule.

| Excitation | Inhibition | Latched after one pulse | Ignites with no pulse | Memory at 5 / 10 / 20 / 40 ms |
|---|---|---|---|---|
| x20 | x3 | 100% | 0% | 38% / 15% / 19% / 29% |
| x20 | x10 | 92% | 0% | 24% / 6% / 16% / 12% |
| x20 | x30 | 28% | 0% | 14% / 6% / 8% / 8% |
| x30 | x3 | 100% | **40%** | 68% / 60% / 47% / 13% |
| x30 | x10 | 100% | 5% | 52% / 36% / 42% / 23% |
| x30 | x30 | 100% | 5% | 21% / 14% / 20% / 17% |

- **Runs?** Yes. **Passes?** No. Inhibition strong enough to stop the latch (x20 with x30) also erases the memory. At x30 excitation, even x30 inhibition doesn't stop the latch. So there's still no setting where activity echoes and then fades.
- **What that suggests:** the latch isn't simply excitation outrunning inhibition. Each excitatory connection has a fast synapse (AMPA, a few ms) and a slow one (NMDA, about 100 ms), and so far both were boosted together. Slow NMDA currents are the textbook mechanism for self-sustaining "on" states. Next: boost only the fast AMPA synapses (in progress).

## What's left to try

1. **Boost only the fast synapses:** AMPA x30–x100 with NMDA left at x1 or removed (`recurrence_sweep.py --nmda`). Same pass rule (in progress).
2. **Longer pulses or pulse trains,** to see whether they drive network activity that lasts.
3. **Real cultures** may have much longer memory than this simulator. That is the biggest open question for Archean.
