# Reservoir Pong

The simulator reflects a falling ball off the side walls of a unit court. Each new cell in a 4×4 grid sends a biphasic pulse to the corresponding physical electrode. A fixed native livn reservoir responds. At each completed count window, a frozen linear Ridge readout maps the latest per-electrode spike counts to a landing estimate.

Raw-pulse Ridge gets the delivered pulse prefix at the same decision times. Follow-the-ball aims at the most recently observed ball position. The same paddle integrator moves each controller, freezes its target at commitment, and scores impact. The software geometry decoder runs during preflight to check that the pulse encoding contains enough information to play.

## Run

From the repository root:

```sh
.venv/bin/python -m tasks.reservoir_pong all
```

`all` runs spatial and memory calibration, software preflight, the pilot, final training, and evaluation. A failed calibration or preflight stops the run before training. Use `--help` for the separate `calibrate`, `preflight`, `pilot`, `train`, `evaluate`, and `serve` commands. Run settings, readouts, trial checkpoints, and reports are saved under `experiments/pong/<run-id>/`. Resume an interrupted evaluation with the same run directory and settings.

```sh
.venv/bin/python -m tasks.reservoir_pong --run-dir experiments/pong/<run-id> serve
```

Open [the local dashboard](http://127.0.0.1:8765/). Run a fresh native simulation or replay a saved trial. Switch controllers on the same ball to compare their estimates, paddle targets, and outcomes. The dashboard can replay the three core traces from older runs and use compatible saved Ridge models for live trials. Older evaluation reports and auxiliary recordings are available as files in their original run directories.

## Read the code

- `physics.py`, `stimulus.py`, `reservoir.py`: ball motion, electrical pulses, and native simulation.
- `experiment.py`, `training.py`, `pipeline.py`: trial generation, shared collection, and Ridge fitting.
- `calibration.py`, `preflight.py`: calibration and task design checks.
- `evaluation_run.py`, `evaluation.py`: held-out evaluation and paired comparisons.
- `web.py`, `dashboard.*`: local API and browser display.

The final report in `final/metrics/evaluation.md` includes the paired opposite-side comparison, unclipped landing error, and controls. The three subreservoirs are spatial selections from one simulated culture. Recording blanking is a simulation rule.
