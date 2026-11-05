"""Pong played by the simulated culture: every frame, the game becomes pulses and the readout steers the paddle.

    python tasks/pong.py            # train, then play every controller on the same balls (~10 min)

=============================================================== PLAN
GAME:        one-sided Pong on a 1 x 1 field. The ball launches from the left
             at a random speed and angle, bounces off the top and bottom walls,
             and scores a hit if the paddle (right wall) covers it on arrival.
INPUT:       what a player sees each frame: the ball now, the ball LAG frames
             ago, and the paddle. Nothing pre-computed (no velocity, no
             landing spot, no "ball relative to paddle").
ENCODING:    the 4 x 4 block of strongest electrodes is the culture's "screen".
             The field is split 4 x 4; a position fires the electrode for its
             zone. Ball LAG frames ago at 20 ms, ball now at 26 ms, paddle (on
             the right-hand column) at 32 ms: 6 ms apart, so responses interact.
DECODING:    spike counts per electrode per 5 ms bin (same as XOR).
READOUT:     a straight line (ridge regression) predicts how far the landing spot
             is from the paddle; the paddle heads that way at full speed. A
             perfect player steers the same way using the true landing spot.
METRIC:      hit rate on the same balls for every controller, against:
             do nothing, follow the ball, a line on the raw positions, a line on
             the same 4 x 4 pulses without neurons, and the perfect player.
===========================================================================
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


# ===========================================================================
# 1. SETTINGS
# ===========================================================================
N_CELLS = 300
AMPLITUDE_UA = 40.0
BIN_MS = 5
TRIAL_MS = 50  # last pulse at 32 ms; activity is over ~5 ms after a pulse

PADDLE_HALF = 0.15  # paddle covers 0.30 of the 1.0-high wall
PADDLE_SPEED = 0.025  # per frame: slower than the ball can climb, so you must anticipate
BALL_VX = (0.03, 0.06)  # per frame, towards the paddle
BALL_VY = (-0.08, 0.08)  # steep enough for bounces
LAG = 4  # the "earlier" snapshot is the ball this many frames ago
ZONES = 4  # field split ZONES x ZONES to match the electrode block

N_TRAIN_STATES = 2000  # frames of gameplay the readout learns from
N_BALLS = 100  # balls every controller plays
WORKERS = 4  # one per fast core on an M4

# The 4 x 4 block of strongest electrodes, laid out as on the array: row 0 is the top.
SCREEN = [[72, 73, 55, 54],
          [76, 77, 51, 50],
          [124, 125, 3, 2],
          [120, 121, 7, 6]]
ONSETS = (20.0, 26.0, 32.0)  # ball LAG frames ago, ball now, paddle

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(DATA_DIR, "results", "pong")


# ===========================================================================
# 2. THE GAME (pure Python, no neurons)
# ===========================================================================
def fold(y):
    """Bounce off walls at 0 and 1."""
    y = np.mod(y, 2.0)
    return 2.0 - y if y > 1.0 else y


# The ball's height is tracked "unfolded" (u keeps growing as if there were no walls)
# and folded into the field only when someone looks at it. That gives correct bounces:
# the direction flips at each wall.
def landing(x, u, vx, vy):
    """Where the ball will cross the paddle wall (x = 1), bounces included."""
    return fold(u + vy * (1.0 - x) / vx)


GAIN = 0.05  # how far off (in field units) before the paddle moves at full speed


def steer(offset):
    """Every controller steers the same way: towards where it thinks the ball will land."""
    return float(np.clip(offset / GAIN, -1.0, 1.0))


def target_offset(ball, vel, p):
    """What the readout learns: how far the landing spot is from the paddle (+ = above)."""
    return landing(*ball, *vel) - p


def perfect_command(ball, vel, p):
    """What a perfect player does: head for the true landing spot."""
    return steer(target_offset(ball, vel, p))


def new_ball(rng):
    """A ball on the left with a random speed and angle, already LAG frames into its flight."""
    vx, vy = rng.uniform(*BALL_VX), rng.uniform(*BALL_VY)
    x, u = 0.02, rng.uniform(0.05, 0.95)
    history = []
    for _ in range(LAG + 1):
        history.append((x, u))
        x, u = x + vx, u + vy
    return {"history": history, "vel": (vx, vy), "p": rng.uniform(0.15, 0.85)}


def observe(game):
    """What the player sees: ball LAG frames ago, ball now, paddle."""
    (xa, ua), (xb, ub) = game["history"][-LAG - 1], game["history"][-1]
    return (xa, fold(ua), xb, fold(ub), game["p"])


def step(game, command):
    """Move the paddle, then the ball. Returns None while in flight, else True (hit) / False (miss)."""
    game["p"] = float(np.clip(game["p"] + np.clip(command, -1, 1) * PADDLE_SPEED, PADDLE_HALF, 1 - PADDLE_HALF))
    x, u = game["history"][-1]
    vx, vy = game["vel"]
    if x + vx >= 1.0:  # arrives this frame
        return abs(landing(x, u, vx, vy) - game["p"]) <= PADDLE_HALF
    game["history"].append((x + vx, u + vy))
    return None


def play(controller, seeds, record=False):
    """Play one ball per seed. Returns hits and (optionally) every frame for replay."""
    hits, traces = [], []
    for s in seeds:
        game = new_ball(np.random.default_rng(s))
        frames = []
        while True:
            obs = observe(game)
            command = controller(obs, game)
            if record:
                frames.append([round(v, 4) for v in obs] + [round(float(command), 3)])
            result = step(game, command)
            if result is not None:
                hits.append(bool(result))
                traces.append({"seed": s, "hit": bool(result), "frames": frames})
                break
    return hits, traces


# ===========================================================================
# 3. ENCODING (game state -> pulses on the 4 x 4 screen)
# ===========================================================================
def zone(v):
    return min(int(v * ZONES), ZONES - 1)


def electrode(x, y):
    """The screen electrode for a position: column from x, row from y (top row = high y)."""
    return SCREEN[ZONES - 1 - zone(y)][zone(x)]


def pulses_for(obs):
    xa, ya, xb, yb, p = obs
    return [(ONSETS[0], electrode(xa, ya)), (ONSETS[1], electrode(xb, yb)), (ONSETS[2], electrode(0.99, p))]


class ScreenEncoding(Encoding):
    amplitude_ua: float = AMPLITUDE_UA
    dt: float = 0.1

    def pulses(self, inputs):
        return pulses_for(inputs)

    def __call__(self, env, t_end, inputs):
        electrodes = np.zeros((int(t_end / self.dt), env.io.num_channels), dtype=np.float32)
        for t, ch in self.pulses(inputs):
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), ch] = self.amplitude_ua
        return env.cell_stimulus(electrodes, dt=self.dt)


def zone_features(obs):
    """The same three pulses as plain on/off flags: the no-neuron version of the encoding."""
    f = np.zeros(3 * 128)
    for k, (_, ch) in enumerate(pulses_for(obs)):
        f[k * 128 + ch] = 1.0
    return f


# ===========================================================================
# 4. DECODING (spikes -> counts per electrode per 5 ms bin)
# ===========================================================================
class BinnedCounts(Decoding):
    duration: int = TRIAL_MS
    bin_ms: int = BIN_MS

    def setup(self, env):
        env.record_spikes()

    def __call__(self, signal, env=None):
        _, times = env.channel_recording(signal.spike_ids, signal.spike_times)
        edges = np.arange(0, self.duration + self.bin_ms, self.bin_ms)
        return {"features": np.concatenate([np.histogram(times.get(ch, []), bins=edges)[0]
                                            for ch in env.io.channel_ids]).astype(np.float32)}


# ===========================================================================
# 5. RESERVOIR (one per worker process; same wiring, own noise)
# ===========================================================================
_LAB = {}


def lab(noise_stream):
    if "env" not in _LAB:
        env = Env(predefined("EI", download_directory=DATA_DIR))
        env.selection(N_CELLS, method="patch")
        env.init().apply_model_defaults()
        env.reseed_noise(stream=noise_stream)
        _LAB.update(env=env, enc=ScreenEncoding(), dec=BinnedCounts())
    return _LAB


def neural_features(obs, noise_stream=0):
    L = lab(noise_stream)
    L["env"].clear()  # fresh trial: back to t = 0, same wiring, new noise
    return L["env"](L["dec"], inputs=obs, encoding=L["enc"])["features"]


def _features_job(job):
    stream, states = job
    return [neural_features(s, stream) for s in states]


def _play_job(job):
    stream, readout, seeds = job

    def controller(obs, game):
        return steer(readout.predict(neural_features(obs, stream)[None])[0])

    return play(controller, seeds, record=True)


# ===========================================================================
# 6. TRAINING DATA: realistic frames, labelled by the perfect player
# ===========================================================================
def training_states(n, rng):
    """Frames from games played by a sloppy perfect player, so states look like real play."""
    states, labels = [], []
    while len(states) < n:
        game = new_ball(rng)
        while len(states) < n:
            obs = observe(game)
            offset = target_offset(game["history"][-1], game["vel"], game["p"])
            states.append(obs)
            labels.append(offset)
            if step(game, steer(offset) + rng.normal(0, 0.7)) is not None:
                break
    return states, np.array(labels)


def parallel(fn, jobs):
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(fn, jobs))


# ===========================================================================
# 7. MAIN: train every readout, then play the same balls with each controller
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    states, y = training_states(N_TRAIN_STATES, rng)
    print(f"{len(states)} training frames; landing spot above the paddle in {np.mean(y > 0):.0%} of them")

    started = time.time()
    chunks = [states[k::WORKERS] for k in range(WORKERS)]
    base = random.randint(10**6, 10**8)
    parts = parallel(_features_job, [(base + k * 100_000, c) for k, c in enumerate(chunks)])
    X_neural = np.zeros((len(states), len(parts[0][0])), dtype=np.float32)
    for k, part in enumerate(parts):
        X_neural[k::WORKERS] = np.array(part)
    print(f"simulated {len(states)} training frames in {(time.time() - started) / 60:.1f} min")

    X_raw = np.array(states)
    X_zone = np.array([zone_features(s) for s in states])
    readouts = {name: Ridge(alpha=10.0).fit(X, y) for name, X in
                [("raw", X_raw), ("zone", X_zone), ("neural", X_neural)]}
    np.savez(os.path.join(OUT_DIR, "training.npz"), X_neural=X_neural, X_raw=X_raw, y=y)

    seeds = list(range(1000, 1000 + N_BALLS))  # the same balls for every controller
    ceiling = HistGradientBoostingRegressor(max_iter=300).fit(X_zone, y)  # best any decoder can do with these pulses
    centre = lambda v: (zone(v) + 0.5) / ZONES
    controllers = {
        "do nothing": lambda obs, g: 0.0,
        "follow the ball, zones only": lambda obs, g: steer(centre(obs[3]) - centre(obs[4])),
        "line on 4x4 pulses, no neurons": lambda obs, g: steer(readouts["zone"].predict(zone_features(obs)[None])[0]),
        "nonlinear on 4x4 pulses (ceiling)": lambda obs, g: steer(ceiling.predict(zone_features(obs)[None])[0]),
        "follow the ball, exact height": lambda obs, g: steer(obs[3] - obs[4]),
        "line on raw positions": lambda obs, g: steer(readouts["raw"].predict(np.array(obs)[None])[0]),
        "perfect player": lambda obs, g: perfect_command(g["history"][-1], g["vel"], g["p"]),
    }
    results, traces = {}, {}
    for name, ctrl in controllers.items():
        hits, tr = play(ctrl, seeds, record=True)
        results[name], traces[name] = float(np.mean(hits)), tr

    started = time.time()
    seed_chunks = [seeds[k::WORKERS] for k in range(WORKERS)]
    base = random.randint(10**6, 10**8)
    played = parallel(_play_job, [(base + k * 100_000, readouts["neural"], c) for k, c in enumerate(seed_chunks)])
    hits = [h for hs, _ in played for h in hs]
    results["neurons + line"] = float(np.mean(hits))
    traces["neurons + line"] = sorted((t for _, ts in played for t in ts), key=lambda t: t["seed"])
    print(f"neurons played {N_BALLS} balls in {(time.time() - started) / 60:.1f} min")

    print(f"\nHit rate over the same {N_BALLS} balls:")
    for name, rate in results.items():
        print(f"  {name:32s} {rate:.0%}")
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"hit_rate": results, "settings": {
            "paddle_half": PADDLE_HALF, "paddle_speed": PADDLE_SPEED, "ball_vx": BALL_VX, "ball_vy": BALL_VY,
            "lag": LAG, "zones": ZONES, "screen": SCREEN, "onsets": ONSETS, "n_train": N_TRAIN_STATES}}, f, indent=2)
    with open(os.path.join(OUT_DIR, "traces.json"), "w") as f:
        json.dump(traces, f)
    print(f"saved results and every frame of every game to {OUT_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
