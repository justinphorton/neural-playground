# Reservoir Pong, runs so far

400 balls, 8x6 court, commit at half height, vx_max 3, ridge readout, 3 train/test splits.
Catch = |prediction - landing| < one cell. Chance ~25%.

| readout                              | catch | bounce | no-bounce | err   |
|--------------------------------------|-------|--------|-----------|-------|
| shuffled labels (control)            | 27%   | 25%    | 30%       | 0.282 |
| baseline: line on last 6 squares     | 31%   | 29%    | 36%       | 0.233 |
| dish, end-of-fall window, 1 pulse    | 30%   | 30%    | 31%       | 0.236 |
| dish, end-of-fall window, 3 pulses   | 31%   | 29%    | 33%       | 0.236 |
| dish, whole fall, 3 pulses           | 32%   | 26%    | 43%       | 0.237 |
| both: squares + whole-fall dish      | 32%   | 27%    | 43%       | 0.232 |

## What was found, in order

1. vx_max 12 (first setting): every linear readout at chance, dish saturated
   (500+ spikes then silence). Too hard for anyone. Dropped to vx_max 3.
2. Spec's "skip blind phase then read" window was empty: dish is exhausted
   right after the last pulse. Read the response to the last pulses instead.
3. Three-path probe (same final square, different approach): same final zap
   gives 87 / 97 / 41 spikes depending on history. Dish is not a mirror.
4. Direction signal was ~0.3 spikes on the column the ball came from. One
   pulse per square barely spills to neighbours. Three pulses per square:
   2.3-spike signal.
5. Whole-fall readout + 3 pulses: dish beats baseline on non-bouncing balls
   (43% vs 36%). Dish encodes velocity from pulse timing.
6. Dish does NOT beat baseline on bouncing balls (26% vs 29%). The bounce
   happens after commit; the dish never sees it, and walls are not encoded.
   The dish can only combine what it is fed.

## Next
Make the ball fast enough to bounce BEFORE commit, so wall contact is in the
dish's state. That tests whether the dish combines "just bounced" with
direction. Also: more balls (400 -> 1500) to tighten the ~3-point error bars.
