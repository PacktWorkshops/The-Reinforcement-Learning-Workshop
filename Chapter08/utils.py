import pandas as pd
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from tqdm import tqdm


class Bandit:
    def __init__(self, optimal_arm_id=0, n_arms=2,
            reward_dists=None, reward_dists_params=None, rand_init=False):
        '''
        When an arm is pulled, a stochastic reward will be drawn from the
        corresponding probability distribution specified in `reward_dists`.
        Each reward distribution is characterized by parameters specified in
        `reward_dists_params`.
        If either of these parameters is `None`, the bandit takes on its default
        form: a 2-arm bandit with Bernoulli(0.6) and Bernoulli(0.4) rewards.
        '''

        if rand_init:
            self.n_arms = n_arms
            self.reward_dists = [np.random.binomial for _ in range(n_arms)]
            self.reward_dists_params = [(1, np.random.rand()) for _ in range(n_arms)]

            self.optimal_arm_id = 0
            for arm_id in range(1, n_arms):
                if self.reward_dists_params[self.optimal_arm_id][1] < self.reward_dists_params[arm_id][1]:
                    self.optimal_arm_id = arm_id

        elif reward_dists is None or reward_dists_params is None:
            self.optimal_arm_id = optimal_arm_id
            self.n_arms = 2
            self.reward_dists = [np.random.binomial for _ in range(n_arms)]
            self.reward_dists_params = [(1, 0.7), (1, 0.3)]

        else:
            self.optimal_arm_id = optimal_arm_id
            self.n_arms = n_arms
            self.reward_dists = reward_dists
            self.reward_dists_params = reward_dists_params

    def pull(self, arm_id):
        if arm_id >= self.n_arms or arm_id < 0 or not isinstance(arm_id, int):
            raise ValueError('Invalid arm index')

        return self.reward_dists[arm_id](*self.reward_dists_params[arm_id])

    def automate(self, policy, n_rounds=100, visualize_regret=False):
        '''
        `policy` should have the following API:
        - a `decide()` method which returns a valid arm index as the next decision.
        - an `update(arm_id, reward)` method which updates its knowledge with
        a newly obtained reward.

        The method returns `history`, the list of pulled arms in order, `rewards`,
        the list of corresponding reward values, and `optimal_rewards`, the list
        of reward values from the genie policy. If `visualize_regret` is `True`,
        the cumulative difference between `rewards` and `optimal_rewards` will
        be plotted as a function of time step,
        '''
        history = []
        rewards = []
        optimal_rewards = []

        rounds = [i for i in range(1, n_rounds + 1)]
        for _ in rounds:
            action = policy.decide()  # get a decision from the input policy
            reward = self.pull(action)  # get a reward from the decision
            policy.update(action, reward)  # update the input policy

            history.append(action)
            rewards.append(reward)
            if action == self.optimal_arm_id:
                optimal_rewards.append(reward)

        # Get the rest of the would-be optimal reward sequence
        while len(optimal_rewards) < n_rounds:
            optimal_rewards.append(self.pull(self.optimal_arm_id))

        # Visualize the cumulative regret
        if visualize_regret:
            plt.plot(
                rounds,
                [
                    cum_optimal_reward - cum_reward
                    for cum_reward, cum_optimal_reward
                    in zip(np.cumsum(rewards), np.cumsum(optimal_rewards))
                ]
            )
            plt.xlabel('Round number')
            plt.ylabel('Cumulative regret')

            plt.show()

        return history, rewards, optimal_rewards

    def repeat(self, policy_class, policy_params,
            n_experiments=100, n_rounds=100, visualize_regret_dist=False):
        total_regrets = []

        for _ in range(n_experiments):
            policy = policy_class(*policy_params)

            history, rewards, optimal_rewards = self.automate(
                policy, n_rounds=n_rounds
            )
            total_regrets.append(np.sum(optimal_rewards) - np.sum(rewards))

        if visualize_regret_dist:
            plt.hist(total_regrets, bins=max(10, n_experiments // 5))
            plt.xlabel(f'Total regrets across {n_experiments} experiments')
            plt.show()

        return total_regrets


class QueueBandit:
    def __init__(self, filename='data.csv', n_classes=3, n_class_customers=100,
            n_experiments=500):
        self.df = pd.read_csv(filename, header=None)
        self.n_classes = n_classes
        self.n_class_customers = n_class_customers
        self.queue_length = n_classes * n_class_customers
        self.n_experiments = n_experiments

    def pull(self, class_, queues):
        queue = queues[class_]
        try:
            job_length = queue[0]
        except KeyError:
            print(queue)
        queues[class_] = queue[1:]

        return job_length, queues

    def automate(self, policy, experiment_id, visualize_cumulative_time=False):
        queue = self.df.iloc[:, experiment_id].to_numpy(copy=True)
        queues = [queue[
            (class_) * self.n_class_customers: (class_ + 1) * self.n_class_customers
        ] for class_ in range(self.n_classes)]
        optimal_order = np.argsort([np.mean(queue) for queue in queues])
        n_customers = self.queue_length

        history = []
        per_waiting_time = []

        rounds = [i for i in range(1, self.queue_length + 1)]
        for _ in rounds:
            action = policy.decide(
                [len(queue) for queue in queues]
            )
            job_length, queues = self.pull(action, queues)
            policy.update(action, job_length)
            n_customers -= 1

            history.append(action)
            per_waiting_time.append(job_length * n_customers)

        if visualize_cumulative_time:
            plt.plot(rounds, np.cumsum(per_waiting_time))
            plt.xlabel('Round number')
            plt.ylabel('Cumulative waiting time')

            plt.show()

        return history, per_waiting_time, optimal_order

    def repeat(self, policy_class, policy_params, experiments='50',
            visualize_cumulative_times=True):
        if experiments == 'all':
            experiments = [i for i in range(self.n_experiments)]
        if experiments == '50':
            experiments = [i for i in range(50)]

        cumulative_times = []
        for experiment_id in tqdm(experiments):
            policy = policy_class(*policy_params)

            history, per_waiting_time, optimal_order = self.automate(
                policy, experiment_id
            )
            cumulative_times.append(np.sum(per_waiting_time))

        if visualize_cumulative_times:
            plt.hist(cumulative_times, bins=max(10, len(experiments) // 5))
            plt.xlabel('Cumulative waiting time')
            plt.show()

        return cumulative_times
