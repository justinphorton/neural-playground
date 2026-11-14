# Reservoir Pong

Reservoir Pong is a standalone livn experiment for studying whether a fixed simulated neural culture helps predict Pong landing positions. Spatial pulses encode ball position on a 4×4 physical electrode grid; the reservoir response is read by a fresh linear model and compared with a matched raw-pulse Ridge baseline.

The implementation lives in [`tasks/reservoir_pong/README.md`](tasks/reservoir_pong/README.md). It includes analytic Pong physics, native livn simulation, label-free calibration, software-only preflight, resumable records, readout fitting, evaluation controls, and a local browser dashboard.

Install the dependencies in your project virtual environment and start a run from the repository root:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m tasks.reservoir_pong all
```

Artifacts are saved under `experiments/pong/<run-id>/`. The pipeline stops before training if label-free calibration or task-design preflight fails. A runnable dashboard is available with `serve`; its live mode uses the same native simulator and its replay mode reads saved records.

Each run records negative findings and gate outcomes. Simulation results are not biological measurements: in particular, the imposed post-pulse blanking interval does not establish amplifier recovery time, and the three deterministic subreservoirs are not independent biological cultures.
