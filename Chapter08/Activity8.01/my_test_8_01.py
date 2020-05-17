import unittest
from ipynb.fs.full.Activity8_01 import GreedyQueue, ETCQueue, ExpThSQueue, \
    ExploitingThSQueue

from utils import QueueBandit


class MyTest(unittest.TestCase):
    def setUp(self):
        self.bandit = QueueBandit(filename='../data.csv')

    def test_greedy(self):
        cumulative_times = self.bandit.repeat(
            GreedyQueue, [3], visualize_cumulative_times=False, experiments=[0, 1]
        )

        # print('greedy')
        # print(cumulative_times)

        self.assertEqual(cumulative_times, [6110.2323977743345, 8955.21419510985])

    def test_etc(self):
        cumulative_times = self.bandit.repeat(
            ETCQueue, [3], visualize_cumulative_times=False, experiments=[0, 1]
        )

        # print('etc')
        # print(cumulative_times)

        self.assertEqual(cumulative_times, [6108.773970488878, 9046.630167012489])

    def test_ths(self):
        cumulative_times = self.bandit.repeat(
            ExpThSQueue, [3], visualize_cumulative_times=False, experiments=[0, 1]
        )

        # print('ths')
        # print(cumulative_times)

        self.assertEqual(cumulative_times, [6202.003285935252, 9204.3616519717])

    def test_mod_ths(self):
        cumulative_times = self.bandit.repeat(
            ExploitingThSQueue, [3], visualize_cumulative_times=False, experiments=[0, 1]
        )

        # print('mod_ths')
        # print(cumulative_times)

        self.assertEqual(cumulative_times, [6377.967408914488, 8984.308594569695])


if __name__ == '__main__':
    unittest.main()
