import unittest

import numpy as np

from tasks.reservoir_pong.permutation import permute_trajectory_targets


class PermutationTests(unittest.TestCase):
    def test_targets_are_permuted_by_trajectory_not_frame(self):
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        targets = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        shuffled = permute_trajectory_targets(targets, groups, np.random.default_rng(12))
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(shuffled[groups == group])), 1)
        self.assertEqual(sorted(np.unique(shuffled)), sorted(np.unique(targets)))

    def test_permutation_rejects_one_trajectory(self):
        with self.assertRaises(ValueError):
            permute_trajectory_targets(np.array([0.2, 0.2]), np.array([1, 1]),
                                       np.random.default_rng(1))

if __name__ == "__main__":
    unittest.main()
