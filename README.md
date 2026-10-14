# neural-playground

A testbed for **neural reservoir computing**: encode an input as electrical stimulation, let a fixed network of neurons transform it, and train a simple readout on the network's response. Everything currently runs in simulation using [livn](https://github.com/livn-org/livn), a simulator of neuron cultures on multi-electrode arrays.

The repo has two parts: experiments (task files) and an interactive web playground that runs the simulation and shows every stage of the pipeline. The playground is the best place to build intuition; this README covers what the project does and how it's implemented.

**Live demo:** the XOR playground is hosted at **https://neural-playground-3bsb.onrender.com/**. Each trial runs a real simulation, so a send takes a few seconds.

![The web playground](docs/playground.png)

## Quick start

Requires Python 3.11+.

```sh
git clone <repo URL>
cd neural-playground

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
python playground/web.py demo_xor
```

This opens the playground at http://localhost:8765. Pick an input, press **Send to the neurons**, and scroll through the steps. Ctrl+C stops the server. The first run downloads the simulated culture into `systems/`.

| Command | What it does |
|---|---|
| `python tasks/demo_xor.py` | Runs the XOR experiment: trains, tests, compares against the baseline, shows spike rasters (1,000 trials one at a time, ~25 min) |
| `python playground/web.py <task>` | Opens the playground for any task in `tasks/` |
| `python playground/web.py <task> --retrain` | Retrains that task's saved readout |

## Approach

```
input → encoding → electrical pulses → neurons (reservoir) → spikes → decoding → features → readout → output
```

The network of neurons, the **reservoir**, is never trained. It's a fixed recurrent system that reacts to whatever it receives. Each trial:

1. **Encode** the input as pulses on the electrodes.
2. **Record** the network's response through the same electrodes.
3. **Decode** the response into a fixed-length feature vector.
4. **Read out** the answer with a linear model trained on those features.

| Fixed | Trained |
|---|---|
| neurons, wiring, synapse strengths | readout weights only |
| encoding (input → pulses) and decoding (spikes → features) | |

The question is whether the reservoir turns the input into a representation a *linear* readout can use. The readout is kept linear deliberately: if a more powerful model solved the task, it wouldn't be clear that the neurons contributed anything. The structure also maps directly onto hardware: encoding is the stimulator, decoding is the recording setup, and the reservoir could be a real culture instead of a simulated one.

## Why XOR first

XOR ("is exactly one of the two bits on?") isn't linearly separable. The 1s and 0s sit on opposite diagonals, so no single straight line separates them, and linear regression fit on the raw bits outputs 0.5 for every input.

![XOR isn't linearly separable](docs/xor_problem.png)

If a linear readout solves XOR after the reservoir, the reservoir must have produced the missing nonlinear feature ("were both bits on?"). The bits are presented **sequentially**, so the network also has to retain a trace of bit 1 when bit 2 arrives. One task tests both nonlinear mixing and short-term memory.

![The same linear regression before and after the reservoir](docs/xor_solved.png)

**Timing is the key design parameter.** The reservoir only helps if bit 2 arrives while the network is still responding to bit 1. Right after a burst, many neurons are refractory and synapses are partly depleted, so the second burst is weaker and involves a different set of electrodes. That difference is the "both bits on" signal. With a 30 ms gap the network recovers completely, the response is just the sum of two independent bursts, and accuracy drops to chance.

| Gap between bits | 3 ms | **6 ms** | 10 ms | 30 ms |
|---|---|---|---|---|
| Accuracy (quick sweep, initial single-electrode setup) | ~75% | **~90%** | ~71% | ~60% |

### Current performance

Each bit fires two neighbouring electrodes (3 and 51) together, and the readout (plain linear regression) is trained on 1,000 trials whose pulses vary by up to ±1.5 ms and 30–50 µA. Tested on fresh trials it never saw (192 per condition):

| Test inputs | Accuracy |
|---|---|
| Normal (bits at 20/26 ms, 40 µA) | **86%** |
| Slightly changed (±0.5 ms, 35 µA) | **90%** |
| Anywhere in the training range (±1.5 ms, 30–50 µA) | **82%** |
| Same linear regression on the raw bits | chance |

Held-out accuracy per input on the training set: (0, 0) 93%, (0, 1) 80%, (1, 0) 86%, (1, 1) 71%. (1, 1) is the hardest case, since it depends on reading the weaker second burst.

The initial version used a single electrode and 100 training trials with exact timing. It scored 75%, 68% and 65% on the same three tests and was fragile: shifting a pulse by 0.5 ms or weakening it by 5 µA could flip the answer. Two electrodes per bit, training variation, and more training trials fixed that. That version is kept in git under the tag `xor-fixed-timing`.

## Implementation

### Simulation (livn)
- **Network:** livn's predefined `EI` culture (excitatory and inhibitory neurons), restricted to a ~300-neuron central patch with `env.selection(300, method="patch")`. A 60 ms trial takes about 1 s; a 200 ms trial on the full ~2,600-neuron culture takes over 30 s.
- **Backend:** livn's `native` backend, a C implementation of its neuron model that matches the NEURON backend spike for spike. It ships precompiled with `pip install livn`.
- **Electrodes:** livn's simulated 128-electrode array. `env.cell_stimulus(...)` maps electrode currents to per-neuron stimulation; `env.channel_recording(...)` groups spikes by electrode.
- **Trials:** one fixed network is reused for every trial. `env.clear()` resets it and draws fresh background noise, so trials are independent and noisy, like repeated recordings of the same culture. Each trial is a single `env(decoding, inputs=..., encoding=...)` call.

### XOR task ([`tasks/demo_xor.py`](tasks/demo_xor.py))
- **Encoding** (`TemporalBitsEncoding`, a livn `Encoding`): each input becomes a list of `(time, electrode)` pulses. A 1 is a 1 ms, 40 µA pulse fired on **two neighbouring electrodes at once**: the electrode that couples most strongly into the patch (3) and its closest strong neighbour (51). Bit 1 fires at 20 ms, bit 2 at 26 ms; a 0 is no pulse. Driving two electrodes recruits more neurons and makes the weaker second burst in (1, 1) easier to read: with a single electrode, (1, 1) was close to chance.
- **Training variation:** training trials jitter each pulse by up to ±1.5 ms and draw its strength from 30–50 µA, so the readout learns which differences matter instead of memorizing one exact pattern. Without it, shifting a pulse by 0.5 ms or weakening it by 5 µA was enough to break the readout.
- **Decoding** (`BinnedChannelCounts`, a livn `Decoding`): spike counts per electrode in 5 ms windows over the 60 ms trial, 128 × 12 = **1,536 features**. Decoding is per electrode, not per neuron, because that's what a real electrode array measures.
- **Readout:** scikit-learn `LinearRegression`, thresholded at 0.5, trained on 1,000 trials (250 per input). Plain least squares needs many trials here: with ~170 informative features and few examples it fits the noise. `demo_xor.py` evaluates with a 70/30 train/test split.
- **Baseline:** the same readout trained on the raw bits.

### Task files ([`tasks/`](tasks))
Each experiment is a single file with nine sections: settings, task definition, data generation, encoding, decoding, reservoir, feature collection, readout and evaluation, main. `demo_xor.py` is the reference implementation. `template_task.py` has the same structure with TODOs and a planning block (task, input, encoding, decoding, hypothesis, metric).

To add an experiment, copy the template into `tasks/`, fill in the plan and TODOs, and run it with `python tasks/my_task.py`. Anything built from the template also runs in the playground (`python playground/web.py my_task`).

### Playground ([`playground/`](playground))
A small web app for exploring a task interactively: `web.py` runs the livn simulation and readout behind a standard-library HTTP server, and `web.html` visualizes each stage of a trial. It trains a task's readout once (in parallel worker processes) and saves it to `playground/saved_<task>.pkl`; the XOR readout is committed so it starts immediately. It's built around the current task-file format (one yes/no task per server, 1 ms pulse encodings, electrode × time-window features), and a task can add its own page text through an optional `PLAYGROUND` dict. Anything beyond that means extending or replacing it.

### Hosting
When the `PORT` environment variable is set (as on Render), `web.py` listens on `0.0.0.0:$PORT` and doesn't open a browser; local behavior is unchanged. For Render: build command `pip install -r requirements.txt`, start command `python playground/web.py demo_xor`, environment variable `PYTHON_VERSION=3.12`. Each trial is a full simulation, so small instances are slow.

### Versions
`requirements.txt` pins exact versions: the committed readout was trained with that scikit-learn version, and livn is pinned to `2.0.0` because its GitHub `main` and documentation are ahead of the PyPI release.

## Next steps

- **More tasks on the same pipeline:** delayed memory, parity, sequence discrimination across electrodes, pulse counting, frequency discrimination, and spatial patterns across electrodes (a toy olfaction task; the fruit-fly mushroom body is itself a fixed random expansion with a simple readout).
- **Readouts:** ridge regression for more stable single-trial predictions, multi-class outputs, and rejection of unfamiliar inputs.
- **Robustness and invariance:** recognizing the XOR pattern at any time, not just at 20/26 ms. With fixed time windows that fails (~53% in tests); lining the windows up with the stimulus onset recovered ~70%. Also systematic sweeps over gap, strength and electrode placement.
- **More realistic recording:** the decoder currently gets exact spike times for every neuron near an electrode. Real electrodes record noisy voltage, need threshold-based spike detection, and are blanked for a few milliseconds after each pulse by the stimulation artifact. livn can simulate the extracellular voltage (`env.potential_recording`), so the gap between idealized and realistic decoding can be measured.
- **Reservoirs:** larger and custom cultures via livn's system generator; the `neuron` backend for large networks and `diffrax` for differentiable simulation (e.g. optimizing the stimulation itself).
- **Beyond fixed reservoirs:** closed-loop control through livn's Gymnasium interface, and synaptic plasticity (`env.enable_plasticity()`).
- **Real cultures:** the encode/decode split mirrors a physical electrode setup, so the same experiment structure should transfer to recordings from real cultures.

The structure will likely change as these are added: task files may become configurations, readouts and decoders swappable components, and experiments tracked systematically.

## Layout

```
neural-playground/
├── README.md
├── requirements.txt
├── tasks/
│   ├── demo_xor.py              XOR experiment (reference)
│   └── template_task.py         starting point for new experiments
├── playground/
│   ├── web.py                   server: livn simulation + readout
│   ├── web.html                 the page
│   └── saved_demo_xor.pkl       trained XOR readout
└── docs/                        README images
```

`systems/` (livn's network data) and `.venv/` are local only and not tracked.
