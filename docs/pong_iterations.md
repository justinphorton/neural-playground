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

## What's left to try

1. **Use the hidden memory on purpose:** pulses that must combine should hit shared neurons, or a probe pulse should read the recovery state.
2. **Longer pulses or pulse trains,** to see whether they drive network activity that lasts.
3. **Real cultures** may have much longer memory than this simulator. That is the biggest open question for Archean.
