import unittest

import numpy as np

from livn.backend.config import backend
from livn.system import System, predefined

from tasks.reservoir_pong.records import ExperimentConfig
from tasks.reservoir_pong.reservoir import (
    ReservoirRunner,
    build_reservoir,
    REPO_ROOT,
    select_electrode_grid_for_cells,
    select_reservoir_gids,
)
from tasks.reservoir_pong.stimulus import PulseEvent


class ReservoirTests(unittest.TestCase):
    def test_three_reservoir_selections_are_disjoint(self):
        self.assertEqual(backend(), "native")
        system = System(predefined("EI", download_directory=str(REPO_ROOT)))
        selections = select_reservoir_gids(system, n_cells=600, count=3)
        self.assertEqual([len(s) for s in selections], [600, 600, 600])
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(set(selections[left]).isdisjoint(selections[right]))

    def test_electrode_grid_tracks_selected_cells_physical_location(self):
        electrodes = np.asarray([
            [row * 8 + col, col * 200.0, row * 200.0, 0.0]
            for row in range(8) for col in range(8)
        ])
        cell_rows = []
        selected = []
        gid = 0
        for cx, cy in ((300.0, 300.0), (1100.0, 1100.0)):
            gids = []
            for dx, dy in ((-20, -20), (-20, 20), (20, -20), (20, 20)):
                cell_rows.append([gid, cx + dx, cy + dy, 0.0])
                gids.append(gid)
                gid += 1
            selected.append(np.asarray(gids, dtype=np.int64))
        cells = np.asarray(cell_rows)
        grid_a = select_electrode_grid_for_cells(electrodes, cells, selected[0], side=4)
        grid_b = select_electrode_grid_for_cells(electrodes, cells, selected[1], side=4)
        self.assertNotEqual([row[0] for row in grid_a], [row[0] for row in grid_b])
        self.assertLess(np.linalg.norm(np.mean(np.asarray(grid_a)[:, 1:3], axis=0) - [300, 300]), 200)
        self.assertLess(np.linalg.norm(np.mean(np.asarray(grid_b)[:, 1:3], axis=0) - [1100, 1100]), 200)

    def test_seeded_trial_replays_exactly_and_plasticity_stays_disabled(self):
        reservoir = build_reservoir(ExperimentConfig(), slot=0)
        try:
            runner = ReservoirRunner(reservoir, ExperimentConfig())
            event = PulseEvent(1.0, int(reservoir.electrode_grid[0][0]), 40.0, 0.1, 0, 0)
            first = runner.simulate([event], duration_ms=25.0, noise_seed=333)
            second = runner.simulate([event], duration_ms=25.0, noise_seed=333)
            self.assertTrue(np.array_equal(first.spike_gids, second.spike_gids))
            self.assertTrue(np.allclose(first.spike_times_ms, second.spike_times_ms))
            self.assertFalse(reservoir.plasticity_enabled)
        finally:
            reservoir.close()

    def test_seeded_whole_and_chunked_simulations_match(self):
        reservoir = build_reservoir(ExperimentConfig(), slot=0)
        try:
            runner = ReservoirRunner(reservoir, ExperimentConfig())
            event = PulseEvent(1.0, int(reservoir.electrode_grid[5][0]), 40.0, 0.1, 1, 1)
            whole = runner.simulate([event], 25.0, 444)
            chunked = runner.simulate([event], 25.0, 444, chunk_ms=5.0)
            self.assertTrue(np.array_equal(whole.spike_gids, chunked.spike_gids))
            self.assertTrue(np.allclose(whole.spike_times_ms, chunked.spike_times_ms))
        finally:
            reservoir.close()


if __name__ == "__main__":
    unittest.main()
