import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing


class Test(unittest.TestCase):
	def setUp(self):
		import Exercise10_01
		self.exercises = Exercise10_01

        self.episodes = 10

	def test_episodes(self):
		self.assertEqual(self.exercises.episodes, self.episodes)

if __name__ == '__main__':
	unittest.main()