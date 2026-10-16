"""Stand-in reservoir for smoke tests when the real culture can't be downloaded.

300 UNCONNECTED cells on a 16-electrode MEA. No wiring, so no bursts and no
memory: results from this are meaningless. It only checks that the code runs.
Not part of any experiment; delete when the real culture is available.
"""

import numpy as np
from livn.env import Env
from livn.io import MEA
from livn.system import ParallelSystem


def build_reservoir(n_cells=300, seed=0):
    mea = MEA()
    ec = np.asarray(mea.electrode_coordinates)
    rng = np.random.default_rng(seed)

    def coords(n):
        xy = rng.uniform(ec[:, 1].min() + 300, ec[:, 1].max() - 300, size=(n, 2))
        z = np.full((n, 1), ec[:, 3].mean())
        return np.hstack([xy, z])

    n_inh = n_cells // 5
    system = ParallelSystem({"EXC": n_cells - n_inh, "INH": n_inh}, coordinates=coords, name="EI")
    env = Env(system, io=mea)
    env.init().apply_model_defaults()
    return env
