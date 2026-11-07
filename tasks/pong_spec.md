# Reservoir Pong (`tasks/pong_spec.py`)

This is a simulation of the team's "Reservoir Pong" task spec. It asks one question: **when a simulated culture of neurons watches a falling Pong ball, can a simple readout of its activity predict where the ball will land, better than the same readout can from the raw input alone?**

The neurons are never trained. Only the readout, a straight line (a weighted sum), learns. So any improvement over the baseline would have to come from the neurons themselves.

## The game

- A ball falls through a box from top to bottom at a fixed speed.
- Each ball also moves sideways at its own random speed, and bounces off the left and right walls.
- At the **commit point**, 60% of the way down, the paddle has to decide where to go. It's too slow to just follow the ball, so it has to predict where the ball will land.
- **Difficulty** is how far sideways the ball can still travel after the commit point: 1 court width (easy), 2 (medium) or 4 (hard). More distance means more bounces still to come, which makes the landing spot harder to predict.

## How a ball becomes electrical pulses

- We use the 16 electrodes that reach our simulated neurons most strongly, arranged as a **4 × 4 grid**. Each grid square stands for one area of the court, and neighbouring areas use neighbouring electrodes.
- Every time the ball moves into a new area, that area's electrode fires **one pulse**. A pulse is very short: half a millisecond negative, then half a millisecond positive (a "biphasic" pulse, like real stimulators use).
- We only send the ball's position. The paddle's position is never sent, and nothing is sent after the commit point.
- The **timing** of the pulses carries the speed: a fast ball crosses areas quickly, so its pulses arrive close together. Each game frame is 2 ms of neuron time.

## What the neurons do, and how we read them

- The simulated culture (livn's "EI" culture, a patch of 300 neurons) reacts to the pulses. Its connections never change.
- We count spikes on every electrode in 5 ms windows over the whole trial. That list of numbers is the neurons' "response."
- A straight line turns that response into one number: the predicted landing position.
- We train on the first 70% of balls and test on the last 30%, so scores come from balls the readout has never seen.

## What we compare against

- **Baseline:** the same straight line, given the raw list of pulses (which area fired at which frame) instead of the neurons' response. The baseline has perfect memory and no noise, so it's a strict comparison.
- **Best decoder:** a flexible nonlinear model on the same inputs. It shows how much a perfect decoder could get from the information available.
- **Controls:** following the ball, shuffling the correct answers (which should score at chance) and predicting the *previous* ball (which should give nothing, since trials don't carry over).

## Results (600 test balls per level; landing side correct)

| | Easy | Medium | Hard |
|---|---|---|---|
| Straight line on the pulses, no neurons (baseline) | 67% | 52% | 51% |
| Best decoder on the pulses | 88% | 89% | 91% |
| **Neurons + straight line** | **66%** | **52%** | **56%** |
| Neurons + straight line, 2 ms blind phase | 53% | 51% | 49% |
| Best decoder on the neurons | 85% | 89% | 86% |

What this means:

- **The task design works.** At medium difficulty, a straight line is close to guessing (52%) while a flexible decoder gets 89%. There's a real gap for the neurons to fill.
- **The neurons don't fill it, in this setup.** With the same straight line, they match the baseline within the noise (about ±4 points). They pass the ball's path along faithfully (a flexible decoder can read 85–89% from them), but they don't reshape it into something a straight line can read better.
- **The controls behave:** shuffled answers score at chance, and there's no carry-over between balls.

## The blind-phase problem

On a real chip, an electrode can't record for a moment after it stimulates (the "blind phase"). The spec builds that in. But in this simulator, **almost the entire response happens in the first millisecond after each pulse.** Throwing away 2 ms after every pulse removes about 98% of the spikes, and the readout falls to chance. So before results from this simulator can tell us anything about the real chip, we need to know how much later, network-driven activity real cultures produce, and either model the artifact or use a culture that responds more slowly.

## Where this differs from the spec

- **4 × 4 grid instead of about 8 × 4:** the 300-neuron patch only has 16 strong electrodes.
- **The blind phase is imposed in software** by deleting spikes, because the simulator has no stimulation artifact. 2 ms is a placeholder, not a measurement.
- **2 ms per game frame and a commit point at 60%** are my choices, since we have no dish measurements yet.
- **Catches are computed from the prediction at the commit point,** not from a live frame-by-frame game.
- **The silenced-dish control isn't run yet.** Turning off connections between neurons would show whether the network or single cells do the work.

## How to run it

```bash
python tasks/pong_spec.py --width 2
```

`--width` is the difficulty (1, 2 or 4). Each run simulates 2,000 balls on 4 CPU cores in about 6–10 minutes. It prints the results table and saves raw spike counts to `results/pong_spec/`, so scores can be recomputed (for example with a different blind phase) without simulating again.

## Related

- `tasks/pong.py` is an earlier version: a closed-loop game where the culture steers the paddle every frame. It also doesn't beat its no-neuron baseline (48% vs 55% hit rate).
