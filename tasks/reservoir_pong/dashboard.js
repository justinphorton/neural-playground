const $ = id => document.getElementById(id);
const names = { neural: "Reservoir Ridge", raw: "Raw-pulse Ridge", follow: "Follow the ball" };
const colors = { neural: "#67d6e5", raw: "#91aaff", follow: "#ffbf69" };
const displayMsPerSimMs = 200;
let setup, trial, cursor = 0, playing = false, frame = 0, lastFrame;

async function api(path, body) {
  const response = await fetch(path, body && {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

const number = (value, digits = 3) => value == null ? "—" : value.toFixed(digits);
const percent = value => value == null ? "—" : number(value * 100, 1) + "%";
const reflect = x => 1 - Math.abs(((x % 2 + 2) % 2) - 1);
const ballX = t => reflect(trial.ball.x0 + trial.ball.vx * t / trial.ball.fall_ms);

function tableRows(id, rows) {
  $(id).replaceChildren(...rows.map(values => {
    const row = document.createElement("tr");
    for (const value of values) row.insertCell().textContent = value;
    return row;
  }));
}

function renderMetrics() {
  const m = setup.metrics;
  $("metrics-table").hidden = !m;
  $("benchmark").textContent = m
    ? `${setup.metrics_stage} · Scientific gate ${m.scientific_success ? "passed" : "did not pass"}. ` +
      `Catch ${percent(m.neural_catch_rate)} vs ${percent(m.raw_catch_rate)}. ` +
      `Landing MAE ${number(m.neural_landing_mae)} vs ${number(m.raw_landing_mae)} court widths.`
    : "No evaluation recorded.";
  tableRows("metrics", Object.entries(m?.reservoirs ?? {}).map(([slot, row]) => [
    slot, percent(row.catch_rate), percent(row.raw_catch_rate), percent(row.opposite_catch_gain),
  ]));
  const controls = m?.matched_test_controls;
  $("controls").textContent = controls
    ? `Zero features ${percent(controls.reservoir_ridge_zero_spike.catch_rate)} · ` +
      `No stimulation ${percent(controls.actual_no_stimulation.catch_rate)}`
    : "";
  const c = setup.calibration;
  const selected = c?.selected;
  $("calibration").textContent = c
    ? `Spatial and paired-pulse memory calibration: ${c.status}.` +
      (selected ? ` ${selected.amplitude_ua} µA, ${selected.phase_ms} ms per phase, ` +
        `${selected.window_ms} ms counting window. Usable gaps ${c.minimum_gap_ms}–${c.maximum_gap_ms} ms.` : "")
    : "No calibration recorded.";
}

function latestIndex(times, time) {
  let i = times.length - 1;
  while (i >= 0 && times[i] > time + 1e-9) i--;
  return i;
}

function controllerAt(key) {
  const i = latestIndex(trial.availability_ms, Math.min(cursor, trial.ball.commit_ms));
  const estimates = trial[key + "_predictions"];
  const paddle = trial.paddle[key];
  const estimate = key === "follow" && i >= 0 && paddle.length
    ? ballX(trial.availability_ms[i]) : estimates?.[i];
  const half = setup.config.paddle_width / 2;
  return {
    estimate,
    target: estimate == null ? null : Math.max(half, Math.min(1 - half, estimate)),
    center: paddle[latestIndex(trial.paddle.times_ms, cursor)],
  };
}

function line(ctx, x0, y0, x1, y1, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

function drawCourt() {
  const canvas = $("court");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  for (let i = 1; i < 4; i++) {
    line(ctx, i * w / 4, 0, i * w / 4, h, "#354a56");
    line(ctx, 0, i * h / 4, w, i * h / 4, "#354a56");
  }
  const commitY = h * trial.ball.commit_ms / trial.ball.fall_ms;
  ctx.setLineDash([6, 4]);
  line(ctx, 0, commitY, w, commitY, "#ffbf69");
  ctx.setLineDash([]);
  ctx.fillStyle = "#ffbf69";
  ctx.font = "14px system-ui";
  ctx.fillText("COMMIT", 8, commitY - 8);
  const y = t => h * t / trial.ball.fall_ms;
  for (let i = 1; i <= 80; i++) {
    const a = cursor * (i - 1) / 80, b = cursor * i / 80;
    line(ctx, w * ballX(a), y(a), w * ballX(b), y(b), "#67d6e5", 2);
  }
  ctx.fillStyle = "#e3edf0";
  ctx.beginPath();
  ctx.arc(w * ballX(cursor), Math.min(h - 4, y(cursor)), 5, 0, Math.PI * 2);
  ctx.fill();
  const key = $("controller").value;
  const { center, target } = controllerAt(key);
  const half = w * setup.config.paddle_width / 2;
  if (center != null) line(ctx, w * center - half, h - 9, w * center + half, h - 9, colors[key], 6);
  if (target != null) line(ctx, w * target, h - 25, w * target, h - 14, colors[key], 2);
}

function drawSpikes() {
  const pulses = trial.pulses.filter(pulse => pulse.time_ms <= cursor + 1e-9);
  const latest = pulses.at(-1);
  const active = new Set(pulses.map(pulse => pulse.electrode_id));
  $("pulse-count").textContent = `${pulses.length} delivered pulses`;
  $("grid").replaceChildren(...trial.grid.map(([channel]) => {
    const cell = document.createElement("span");
    cell.textContent = `ch ${channel}`;
    cell.classList.toggle("pulsed", active.has(channel));
    cell.classList.toggle("latest", latest?.electrode_id === channel);
    return cell;
  }));
  const canvas = $("raster");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const x = t => w * t / trial.ball.fall_ms;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#6d5024";
  for (const pulse of pulses) {
    const end = Math.min(cursor, pulse.time_ms + trial.metadata.feature_blank_ms);
    ctx.fillRect(x(pulse.time_ms), 0, Math.max(1, x(end - pulse.time_ms)), h);
  }
  const channels = trial.metadata.recording_channels ?? [...new Set(trial.spikes.map(s => s.channel))];
  const rows = new Map(channels.map((channel, i) => [channel, (i + 0.5) * h / channels.length]));
  ctx.fillStyle = "#67d6e5";
  for (const spike of trial.spikes) {
    const t = spike.time_ms - trial.metadata.warmup_ms;
    const y = rows.get(spike.channel);
    if (t >= -1e-9 && t <= cursor + 1e-9 && y != null) ctx.fillRect(x(t), y - 2, 2, 4);
  }
  const i = latestIndex(trial.availability_ms, cursor);
  const nonzero = trial.features[i]?.flatMap((count, j) => count ? [`ch ${channels[j]}: ${count}`] : []);
  $("features").textContent = i < 0 ? "Waiting for the first count window"
    : `Window at ${number(trial.availability_ms[i], 1)} ms · ${nonzero?.join(", ") || "zero spikes"}`;
}

function renderControllerResults() {
  const atImpact = cursor >= trial.ball.fall_ms - 1e-9;
  const outcome = key => !atImpact ? "—" : ({ true: "Caught", false: "Missed" })[trial.outcomes[key]] ?? "Unscored";
  tableRows("comparison", Object.keys(names).map(key => {
    const { estimate, target, center } = controllerAt(key);
    return [names[key], number(estimate), number(target), number(center), outcome(key)];
  }));
  $("decision").textContent = cursor >= trial.ball.commit_ms - 1e-9
    ? `Target locked at ${number(trial.ball.commit_ms, 1)} ms.` : "Target can update until commitment.";
  const key = $("controller").value;
  $("impact").textContent = atImpact
    ? `Landing ${number(trial.ball.landing_x)} · ${names[key]} · ${outcome(key)}` : "";
}

function renderTrial() {
  if (!trial) return;
  $("scrub").value = Math.round(1000 * cursor / trial.ball.fall_ms);
  $("clock").textContent = `${number(cursor, 1)} / ${number(trial.ball.fall_ms, 1)} ms`;
  drawCourt();
  drawSpikes();
  renderControllerResults();
}

function pause() {
  playing = false;
  cancelAnimationFrame(frame);
  $("play").textContent = "Play";
}

function play() {
  if (!trial) return;
  if (cursor >= trial.ball.fall_ms) cursor = 0;
  playing = true;
  lastFrame = null;
  $("play").textContent = "Pause";
  function step(now) {
    if (!playing) return;
    const elapsed = now - (lastFrame ?? now);
    lastFrame = now;
    cursor = Math.min(trial.ball.fall_ms, cursor + elapsed * Number($("speed").value) / displayMsPerSimMs);
    renderTrial();
    if (cursor < trial.ball.fall_ms) frame = requestAnimationFrame(step);
    else pause();
  }
  frame = requestAnimationFrame(step);
}

function updateSaved(ids, selected) {
  $("saved").replaceChildren(...ids.map(id => new Option(id, id)));
  $("saved").value = ids.includes(selected) ? selected : (ids.at(-1) ?? "");
  $("replay").disabled = ids.length === 0;
}

async function loadTrial(id, live = false) {
  pause();
  trial = await api("/api/trials/" + encodeURIComponent(id));
  cursor = 0;
  $("status").textContent = `${live ? "Live simulation" : "Recorded replay"} · ${id}`;
  if (trial.reservoir_slot != null) $("slot").value = trial.reservoir_slot;
  localStorage.setItem(setup.run_id, id);
  renderTrial();
  if (live) play();
}

async function liveTrial() {
  let job = await api("/api/trials", { mode: "live", slot: Number($("slot").value) });
  while (job.status === "queued" || job.status === "running") {
    $("status").textContent = job.progress;
    await new Promise(resolve => setTimeout(resolve, 200));
    job = await api("/api/jobs/" + job.id);
  }
  if (job.status !== "complete") throw new Error(job.error || "Simulation did not finish");
  const ids = [...Array.from($("saved").options, option => option.value), job.trial_id];
  updateSaved(ids, job.trial_id);
  await loadTrial(job.trial_id, true);
}

async function perform(action) {
  $("live").disabled = $("replay").disabled = true;
  try {
    await action();
  } catch (error) {
    $("status").textContent = error.message;
  } finally {
    $("live").disabled = !setup;
    $("replay").disabled = !$("saved").value;
  }
}

async function init() {
  setup = await api("/api/setup");
  $("run").textContent = setup.run_id;
  for (let i = 0; i < setup.config.reservoir_count; i++) $("slot").add(new Option(i, i));
  updateSaved(setup.trials, localStorage.getItem(setup.run_id));
  renderMetrics();
  if ($("saved").value) await loadTrial($("saved").value);
  else $("status").textContent = "Run a live trial to see the simulation.";
}

$("live").onclick = () => perform(liveTrial);
$("replay").onclick = () => perform(() => loadTrial($("saved").value));
$("controller").onchange = renderTrial;
$("play").onclick = () => playing ? pause() : play();
$("scrub").oninput = () => {
  pause();
  if (!trial) return;
  cursor = trial.ball.fall_ms * Number($("scrub").value) / 1000;
  renderTrial();
};
perform(init);
