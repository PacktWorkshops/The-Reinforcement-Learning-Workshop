import unittest
from ipynb.fs.full.The_e_Greedy_Algorithm import eGreedy

import numpy as np
np.random.seed(0)

from utils import Bandit


class MyTest(unittest.TestCase):
    def test_e_greedy(self):
        bandit = Bandit(
            optimal_arm_id=0,
            n_arms=3,
            reward_dists=[np.random.binomial for _ in range(3)],
            reward_dists_params=[(1, 0.9), (1, 0.8), (1, 0.7)]
        )

        egreedy_policy = eGreedy(n_arms=3)

        history, rewards, optimal_rewards = bandit.automate(
            egreedy_policy, n_rounds=10, visualize_regret=False
        )

        self.assertEqual(history, [0, 1, 2, 1, 2, 1, 0, 0, 1, 2])
        self.assertEqual(rewards, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

if __name__ == '__main__':
    unittest.main()
