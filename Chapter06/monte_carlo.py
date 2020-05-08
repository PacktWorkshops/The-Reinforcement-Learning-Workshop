# importing standard libraries
import numpy as np
import gym


def initialize_environment():
    """initialize the OpenAI Gym environment"""
    env = gym.make("FrozenLake-v0", is_slippery=False)
    print("Initializing environment")
    # reset the current environment
    env.reset()
    # show the size of the action space
    action_size = env.action_space.n
    print(f"Action space: {action_size}")
    # Number of possible states
    state_size = env.observation_space.n
    print(f"State space: {state_size}")
    return env


def play(policy, render=False):
    """
    Perform a test pass on the Taxi-v3 environment

    Args:
        policy: the policy to use
        render: if the result should be rendered at every step. False by default

    """
    env = initialize_environment()
    rewards = []
    # max number of steps the agent is allowed to take. If it doesn't reach
    # a solution in this time then we call it an episode and proceed
    max_steps = 25
    test_episodes = 50
    for episode in range(test_episodes):
        # reset the environment every new episode
        state = env.reset()
        total_rewards = 0
        print("*" * 100)
        print("Episode {}".format(episode))

        for step in range(max_steps):
            # Take action which has the highest q value
            # in the current state
            action = policy[state]
            new_state, reward, done, info = env.step(action)
            if render:
                env.render()
            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                print("Score", total_rewards)
                break
            state = new_state
    env.close()
    print("Average Score", sum(rewards) / test_episodes)


def first_visit_monte_carlo(env: gym.Env):
    V = dict()
    # initially the value function for all states
    # will be random values close to zero
    state_size = env.observation_space.n
    # initially we will start with a random policy
    current_policy = dict()
    action_size = env.action_space.n
    episodes = 10000
    current_ep = 0
    Q = np.zeros((state_size, action_size))
    N = np.ones((state_size, action_size))
    gamma = 0.9
    while current_ep < episodes:
        # create a random policy
        current_ep += 1
        for s in range(state_size):
            current_policy[s] = env.action_space.sample()
        state = env.reset()
        done = False
        history = []
        total_reward = 0
        while not done:
            if np.random.randn() < 0.2:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            new_state, reward, done, info = env.step(action)
            # env.render()
            total_reward += reward
            history.append((state, action))

            # if we go out of environment
            if state == new_state and reward <= 0:
                total_reward -= 1
                done = True
            # if we win!
            if done and reward > 0:
                total_reward += 10

            state = new_state
        G = total_reward
        for (state, action) in reversed(history):
            G = gamma * G
            # print(f"State: {state} Action: {action} G: {G}")
            Q[state][action] += G
            N[state][action] += 1

    Q /= N
    print("*" * 30)
    print("Q**********")
    print(Q)
    print("N**********")
    print(N)
    return Q


def create_policy(Q, env: gym.Env):
    current_policy = dict()
    for s in range(env.observation_space.n):
        current_policy[s] = np.argmax(Q[s, :])
    return current_policy


if __name__ == '__main__':
    env = initialize_environment()
    Q = first_visit_monte_carlo(env)
    print(Q.shape)
    policy = create_policy(Q, env)
    print(policy)
    play(policy)
