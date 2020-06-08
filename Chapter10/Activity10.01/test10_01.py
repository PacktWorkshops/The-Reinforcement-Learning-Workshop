import unittest
import import_ipynb
import pandas as pd
import pandas.testing as pd_testing
import numpy.testing as np_testing
import gym
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
import datetime


class DQN():
	def __init__(self, env, batch_size=64, max_experiences=5000):
		self.env = env
		self.input_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.max_experiences = max_experiences
		self.memory = deque(maxlen=self.max_experiences)
		self.batch_size = batch_size
		self.gamma = 1.0
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = self.build_model()
		self.target_model = self.build_model()

	def build_model(self):
		model = Sequential()
		model.add(Conv2D(32, 8, (4, 4), activation='relu', padding='valid', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)))
		model.add(Conv2D(64, 4, (2, 2), activation='relu', padding='valid'))
		model.add(Conv2D(64, 3, (1, 1), activation='relu', padding='valid'))
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dense(self.action_size))
		model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, epsilon=self.epsilon_min), metrics='accuracy')

		return model

	def get_action(self, state):
		if np.random.random() <= self.epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.model.predict(tf.expand_dims(state, 0)))

	def add_experience(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
		self.update_epsilon()

	def replay(self, episode):
		x_batch, y_batch = [], []
		minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

		for state, action, reward, next_state, done in minibatch:
			y_target = self.target_model.predict(tf.expand_dims(state, 0))
			y_target[0][action] = reward if done else reward + self.gamma * np.max(
				self.model.predict(tf.expand_dims(next_state, 0))[0])
			x_batch.append(state)
			y_batch.append(y_target[0])

		self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

	def update_epsilon(self):
		self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)


class Test(unittest.TestCase):
	def setUp(self):
		import Activity10_1
		self.exercises = Activity10_1

		np.random.seed(168)
		tf.random.set_seed(168)

		self.env = gym.make('BreakoutDeterministic-v4')
		self.IMG_SIZE = 84
		self.SEQUENCE = 4
		self.agent = DQN(self.env)
		self.episodes = 50

	def test_IMG_SIZE(self):
		self.assertEqual(self.exercises.IMG_SIZE, self.IMG_SIZE)

	def test_SEQUENCE(self):
		self.assertEqual(self.exercises.SEQUENCE, self.SEQUENCE)

	def test_episodes(self):
		self.assertEqual(self.exercises.episodes, self.episodes)

	def test_model_summary(self):
		self.assertEqual(self.exercises.agent.model.summary(), self.agent.model.summary())

	def test_target_model_summary(self):
		self.assertEqual(self.exercises.agent.target_model.summary(), self.agent.target_model.summary())


if __name__ == '__main__':
	unittest.main()