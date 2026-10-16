# Fixed-Reservoir C4-vs-E4 Memory Experiment

## Current status

The complete simulation-to-website pipeline exists and runs. It contains a frozen neural reservoir, an audited C4/E4 audio dataset, 1,104 saved simulations, extracted neural features, a trained scikit-learn classifier, and a local WAV-upload website.

The main scientific result is negative: the current eight-electrode readout does not predict C4 versus E4 reliably after the sound has stopped. Its grouped cross-validation result is close to the 50% chance level, and real uploads have also produced incorrect predictions. The saved classifier should therefore be presented as a **development prototype**, not as a working piano-note detector or evidence of neural memory.

## Research question

The experiment asks:

> After a one-second C4 or E4 stimulus ends and the applied current returns to exactly zero, can the preceding note be decoded from the reservoir's later spikes alone?

This is specifically a post-stimulus memory test. It is not intended to classify the audio waveform directly.

## Data used

The current C4/E4 manifest contains 256 WAV files:

| Item | Count |
|---|---:|
| C4 WAVs | 128 |
| E4 WAVs | 128 |
| Paired C4/E4 conditions | 128 |
| Piano families | 2 |
| Velocity values | 16 |
| Hold durations | 4 |

The two piano families are `AcousticGrand` and `BrightAcoustic`. The hold durations are 250, 600, 1100, and 1800 ms. Each condition group pairs one C4 and one E4 with the same piano family, velocity, and hold duration.

There are also 128 D4 WAVs in the repository. They were deliberately preserved but are excluded from this two-class experiment and from the trained model.

This dataset does **not** establish generalization to arbitrary pianos. Both current piano families came from the same existing dataset structure, and all 256 C4/E4 WAVs were used to fit the final development model.

## Audio preprocessing and stimulation

Every WAV goes through the same preprocessing:

1. Read the WAV and downmix stereo to mono if necessary.
2. Detect the beginning of the active note.
3. Extract a one-second waveform.
4. Resample it to 10,000 Hz.
5. Remove the DC offset.
6. Peak-normalize the waveform.
7. Apply a 15 ms raised-cosine fade at the end.
8. Force the last sample to exactly zero.
9. Multiply the normalized waveform by the selected peak current.

Each simulation lasts 2,500 ms:

| Phase | Absolute time | Applied current |
|---|---|---|
| Baseline | 0–500 ms | Exactly zero |
| Stimulation | 500–1500 ms | One-second piano waveform |
| Memory/readout | 1500–2500 ms | Exactly zero |

The whole 2.5-second sequence is run continuously so the reservoir state is not reset between phases.

The audio audit found a small class-associated RMS difference after normalization: mean normalized RMS was 0.2261 for C4 and 0.2130 for E4. That is a possible nuisance variable, although audio statistics are never passed directly to the neural classifier.

## Fixed neural reservoir

The experiment uses one canonical livn reservoir:

| Setting | Value |
|---|---|
| livn version | 2.0.0 |
| Backend | ctypes |
| Selected neurons | 300 |
| Structural/simulation seed | 123 |
| Neuron model | ReducedCalciumSomaDendrite |
| Plasticity | Disabled |
| Stimulation electrode | 3 |
| Recording electrodes | 121, 7, 6, 125, 2, 77, 51, 50 |
| Reservoir hash | `bbfc20d53052c76b869dba3c0897184c9c24c51321dff4fc2c73d9981f8da9c0` |

Eight independent worker processes loaded the artifact during preflight and returned the same reservoir hash. Each trial also records the structural hash before and after simulation. All 1,104 valid raw trials reference the same canonical hash, and all 1,104 noise seeds are unique.

## What is recorded in each simulation

Each raw trial is stored as a compressed, non-pickle `.npz` file. It contains:

- every recorded spike time and neuron ID from all 300 simulated neurons;
- spike-event times from the eight selected recording electrodes;
- the recording-electrode/channel ID for each event;
- the C4, E4, or control label;
- recording ID, condition group, repeat number, and unique noise seed;
- reservoir, configuration, stimulus, and WAV provenance hashes;
- the pre- and post-trial reservoir hashes;
- current peak/RMS and end-of-stimulus diagnostics;
- phase durations, electrode identities, fade duration, and software versions;
- audio preprocessing/audit values.

The trained classifier does not receive the 300-neuron arrays. It receives only features made from the eight recording-electrode event streams. It also does not receive the WAV, filename, note label, current waveform, current statistics, audio RMS, or stimulation-electrode data.

## Simulations completed

There are 1,104 valid raw trial files:

| Peak current | C4 | E4 | Control | Total |
|---:|---:|---:|---:|---:|
| 0 µA | 0 | 0 | 48 | 48 |
| 1.5 µA | 48 | 48 | 0 | 96 |
| 3.125 µA | 384 | 384 | 0 | 768 |
| 6.25 µA | 24 | 24 | 0 | 48 |
| 12.5 µA | 24 | 24 | 0 | 48 |
| 25 µA | 48 | 48 | 0 | 96 |
| **Total** | **528** | **528** | **48** | **1,104** |

The 768 trials at 3.125 µA are the full training collection: 256 WAVs × 3 independent noise realizations. The other 336 note trials and 48 controls belong to the amplitude screen.

The screen showed that stronger current greatly increased firing during stimulation, but it did not create a clear post-stimulus C4/E4 signal. The recorded during-stimulus rates ranged from about 24.6 at 1.5 µA to 501.2 at 25 µA, while post-stimulus rates returned to roughly 24–25. The screen selected 3.125 µA for the full run.

## Neural features used by the model

Nineteen separate 50 ms post-stimulus windows were evaluated, starting at 5 ms after stimulus offset and ending at 955 ms.

The saved classifier uses the 205–255 ms post-stimulus window and `fine_temporal` features:

- 50 ms observation period;
- divided into ten 5 ms bins;
- spike counts from eight recording electrodes in every bin;
- 10 bins × 8 electrodes = **80 input features per simulated trial**.

The baseline phase and stimulation phase are not included in the note-memory model. The chosen observation window begins 205 ms after current has returned to zero.

## Prediction model

The predictor is a scikit-learn pipeline:

1. `StandardScaler` standardizes each of the 80 neural features.
2. L2-regularized `LogisticRegression` predicts C4 versus E4.

Exact logistic-regression settings:

```text
C = 1.0
penalty = L2
class_weight = balanced
max_iter = 2000
random_state = 0
```

During five-fold grouped cross-validation, the scaler and logistic regression are fit only on each training fold. All three repeats of a WAV stay together, and the paired C4/E4 condition group stays together. After development evaluation, an all-data version of the same pipeline was fit to all 768 simulated responses and saved for the website.

For an uploaded WAV, the website runs three new stochastic reservoir simulations, calculates one C4/E4 probability pair for each simulation, and averages the three probability pairs. The correct note selected by the user is used only to mark the result as a match or mismatch; it is not supplied to the classifier.

## Official development result

For the selected 205–255 ms window:

| Metric | Simulation level | Recording level, three repeats averaged |
|---|---:|---:|
| Accuracy | 53.52% | 51.17% |
| Balanced accuracy | 53.52% | 51.17% |
| ROC AUC | 51.45% | 51.68% |

Simulation-level confusion matrix, with rows as true `[C4, E4]` and columns as predicted `[C4, E4]`:

```text
[[161, 223],
 [134, 250]]
```

Simulation-level recall was 41.93% for C4 and 65.10% for E4, showing a noticeable tendency to predict E4. After averaging the three repeats per WAV, the confusion matrix was:

```text
[[66, 62],
 [63, 65]]
```

This corresponds to 51.56% C4 recall and 50.78% E4 recall. In practical terms, it is approximately coin-flip performance.

The selected window had an uncorrected 100-permutation p-value of 0.0297. However, 19 time windows were searched. The planned Holm-Bonferroni correction, confidence-interval gate, and 1,000-permutation final analysis were not completed by the current post-processing script. The apparent uncorrected result does not survive a conservative 19-window correction, so it should **not** be presented as statistically supported neural memory.

## Exploratory all-neuron result

As a post-hoc diagnostic, the same trials were re-read using spike counts from all 300 simulated neurons between 5 and 505 ms after offset. This was not the preregistered eight-electrode model.

That exploratory readout reached 55.60% simulation-level balanced accuracy and 60.16% recording-level balanced accuracy, with recording-level ROC AUC 0.618. This suggests there may be weak note-related information somewhere inside the simulated reservoir that is lost when activity is projected onto only eight electrode streams.

This is not a validated positive result: the feature choice was made after examining the data, it has not passed the full corrected statistical analysis, and it is not the model installed in the website.

## What worked

- A reproducible C4/E4 manifest was created with stable file hashes and paired nuisance conditions.
- D4 was excluded from this experiment without deleting its source WAVs.
- Audio preprocessing, onset detection, resampling, DC removal, normalization, tapering, and current limits are implemented and audited.
- A single canonical 300-neuron reservoir artifact is loaded by all workers.
- Plasticity is disabled and the reservoir identity is checked before and after trials.
- Baseline, stimulation, and post-stimulus phases run continuously in one simulation.
- 1,104 raw simulations completed with unique noise seeds and one reservoir hash.
- The 768-trial full C4/E4 training collection is complete.
- Neural-only feature extraction and fold-local standardization are implemented.
- A trained model and provenance metadata were saved.
- A local website accepts WAV uploads, runs the reservoir, shows probabilities, and compares the guess with a user-supplied correct note.
- The current automated test suite passes: 24 tests, 0 failures.

## What did not work or remains incomplete

### Main scientific failure

The current eight-electrode reservoir features do not separate C4 and E4 reliably after stimulation. Recording-level balanced accuracy is 51.17%, essentially chance. The user's real website tests also exposed incorrect predictions, which agrees with the cross-validation result.

### Development selection is optimistic

The model was chosen by searching 19 post-stimulus windows and selecting the best window using an uncorrected p-value. The current script does not apply the planned Holm correction, 2,000 grouped bootstraps, 1,000 final permutations, or the contiguous memory-horizon rule before saving a model.

### Amplitude-screen gate is weaker than specified

The code comment says a stimulation response should exceed control by two standard-error units, but the implementation only checks whether the mean response is numerically greater than control. At 3.125 µA, the during-stimulus rate was 25.65 versus 24.96 for controls, a small difference. The saved screen report also says the second-pass amplitudes were 6.25 and 25 µA, while the actual saved schedule contains second repeats for 1.5 and 25 µA. This screen should be corrected before another expensive run.

### External generalization is untested

No genuinely new piano family has been held out. Because the final model was fit on every current C4/E4 WAV, only future WAVs from new sources can test generalization honestly.

### The report is incomplete

The `plots/` directory is empty. The planned PCA figures, accuracy-over-time confidence bands, response-distance curves, corrected significance visualization, and concise generated scientific report were not produced.

### Raw-trial sidecar cleanup issue

All 1,104 `.npz` files are valid and contain their metadata internally. However, only three normal JSON sidecars exist; 1,101 hidden temporary JSON sidecars remain from the parallel collection path. They are not used by model training, but this is a storage/atomic-write bug that should be fixed and cleaned before calling the artifact format finished.

### Representative current arrays were not saved

Every raw trial stores the stimulus hash and scalar current diagnostics, but the experiment folder does not contain the planned representative full current-array files for every screened amplitude.

## What can honestly be claimed

The project successfully demonstrates an end-to-end, reproducible experimental pipeline:

> Piano WAVs were converted to bounded electrical stimuli, passed through one frozen 300-neuron livn reservoir, recorded through eight electrode channels, converted into post-stimulus spike-count features, and decoded with a standardized logistic-regression readout.

It also produced a useful negative result:

> Under the current direct-waveform encoding, fixed reservoir, 3.125 µA amplitude, eight-electrode readout, and 205–255 ms observation window, C4-versus-E4 decoding was approximately chance and did not provide statistically credible evidence of post-stimulus note memory.

It cannot yet claim accurate note prediction, robust memory, or generalization to new pianos.

## Where everything is stored

Repository root:

```text
/Users/aviksolanki/Projects/neural-playground
```

Current experiment:

```text
/Users/aviksolanki/Projects/neural-playground/experiments/
  20261015T074148Z_fixed-c4-e4-memory_22d5b1c13358/
```

Important locations:

| Content | Location |
|---|---|
| C4 source WAVs | `tasks/piano_notes_task/training_data/C4/` |
| E4 source WAVs | `tasks/piano_notes_task/training_data/E4/` |
| Preserved D4 WAVs | `tasks/piano_notes_task/training_data/D4/` |
| Experiment code | `tasks/piano_notes_task/` |
| Tests | `tasks/piano_notes_task/tests/` |
| Experiment configuration | `experiments/.../config.json` |
| 256-file training manifest | `experiments/.../dataset_manifest.csv` |
| Audio audit | `experiments/.../audio_audit.csv` |
| Frozen reservoir | `experiments/.../reservoir/` |
| All 1,104 raw trials | `experiments/.../raw_trials/` |
| Nineteen feature matrices | `experiments/.../features/` |
| Screen schedule/results | `experiments/.../metrics/screen_schedule.json` and `screen_analysis_final.json` |
| Development metrics | `experiments/.../metrics/development_full_metrics.json` |
| Trained model | `experiments/.../models/fixed_readout.joblib` |
| Model provenance | `experiments/.../models/fixed_readout.joblib.json` |
| Local web application | `tasks/piano_notes_task/piano_notes_web.py` |

The experiment folder currently occupies about 17 MB. Most of it is the 15 MB `raw_trials/` directory.

## How to run the demonstration website

From the repository root:

```bash
cd /Users/aviksolanki/Projects/neural-playground
.venv/bin/python -m tasks.piano_notes_task.piano_notes_web \
  --experiment experiments/20261015T074148Z_fixed-c4-e4-memory_22d5b1c13358 \
  --host 127.0.0.1 \
  --port 8765 \
  --repeats 3
```

Then open:

```text
http://127.0.0.1:8765
```

The page will accept a WAV, ask for the known C4/E4 label, run three reservoir simulations, show the average class probabilities, and report whether the model's guess matched the supplied label.

## Recommended next experiment

Before running another 768 trials:

1. Fix the amplitude gate and the mismatch between the screen schedule and report.
2. Add the full corrected statistical report and recording-level model-selection logic.
3. Pilot a label-blind audio-to-pulse encoding instead of sending the complete 10 kHz waveform directly.
4. Test more recording channels, or use the simulated 300-neuron state as an explicit comparison condition.
5. Screen candidate reservoir seeds using dynamical diagnostics, then freeze one before examining note labels.
6. Use firing-rate state vectors at fixed offsets and compare logistic regression with a simple ridge readout.
7. Require a small pilot to beat a paired-label null before committing to another full collection.
8. After freezing the entire pipeline, evaluate genuinely new piano recordings without retraining.

## Reproducibility identifiers

```text
Experiment ID: 20261015T074148Z_fixed-c4-e4-memory_22d5b1c13358
Config hash: 22d5b1c133581b81525c306c31ec65768544733c8bd27b8e8bf55e0096907bd1
Reservoir hash: bbfc20d53052c76b869dba3c0897184c9c24c51321dff4fc2c73d9981f8da9c0
Model status: development
Selected amplitude: 3.125 µA
Selected window: 205–255 ms after stimulus offset
Feature mode: fine_temporal
Model classes: C4, E4
```
