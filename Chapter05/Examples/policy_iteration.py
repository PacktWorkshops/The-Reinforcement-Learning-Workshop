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


def policy_evaluation(V, current_policy,
                      env, gamma, small_change):
    """
    Perform policy evaluation iterations until the smallest change is less than
    `smallest_change`

    Args:
        V: the value function table
        current_policy: current policy
        env: the OpenAI Tax-v3 environment
        gamma: future reward coefficient
        small_change: how small should the change be for the iterations to stop

    Returns:
        V: the value function after convergence of the evaluation
    """
    state_size = env.observation_space.n

    while True:
        biggest_change = 0
        # loop through every state present
        for state in range(state_size):
            old_V = V[state]
            # take the action according to the current policy
            action = current_policy[state]
            prob, new_state, reward, done = env.env.P[state][action][0]
            # use the bellman optimality equation to update V(s)
            V[state] = reward + gamma * V[new_state]
            # if the biggest change is small enough then it means
            # the policy has converged, so stop.
            biggest_change = max(biggest_change, abs(V[state] - old_V))
        if biggest_change < small_change:
            break
    return V


def policy_improvement(V, current_policy,
                       env, gamma):
    """
    Perform policy improvement using the Bellman Optimality Equation.

    Args:
        V: the value function table
        current_policy: current policy
        env: the OpenAI Tax-v3 environment
        gamma: future reward coefficient

    Returns:
        current_policy: the updated policy
        policy_changed: True, if the policy was changed, else, False
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n
    policy_changed = False
    for state in range(state_size):
        best_val = -np.inf
        best_action = -1
        # loop over all actions and select the best one
        for action in range(action_size):
            prob, new_state, reward, done = env.env.P[state][action][0]
            # calculate the future reward by taking this action
            # note: we're using simplified equation because we don't have
            # non-one transition probabilities
            future_reward = reward + gamma * V[new_state]
            if future_reward > best_val:
                best_val = future_reward
                best_action = action
        # using assert statements we can avoid getting into unwanted
        # situations
        assert best_action != -1
        if current_policy[state] != best_action:
            policy_changed = True
        # update the best action for this current state
        current_policy[state] = best_action
    # if the policy didn't change, it means we have converged
    return current_policy, policy_changed


def policy_iteration(env):
    """
    Find the most optimal policy for the Taxi-v3 environment using Policy
    Iteration

    Args:
        env: Taxi=v3 environment

    Returns:
        policy: the most optimal policy
    """
    V = dict()
    # initially the value function for all states
    # will be random values close to zero
    state_size = env.observation_space.n
    for i in range(state_size):
        V[i] = np.random.random()

    # when the change is smaller than this, stop
    small_change = 1e-20
    # future reward coefficient
    gamma = 0.9
    episodes = 0
    # train for this many episodes
    max_episodes = 50000

    # initially we will start with a random policy
    current_policy = dict()
    for s in range(state_size):
        current_policy[s] = env.action_space.sample()

    while episodes < max_episodes:
        episodes += 1
        # policy evaluation
        V = policy_evaluation(V, current_policy,
                              env, gamma, small_change)
        # policy improvement
        current_policy, policy_changed = policy_improvement(V, current_policy,
                                                            env, gamma)
        # if the policy didn't change, it means we have converged
        if not policy_changed:
            break
    print(f"Number of episodes trained: {episodes}")
    return current_policy


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


def random_step(n_steps=5):
    """
    Steps through the taxi v3 environment randomly
    Args:
        n_steps: Number of steps to step through
    """
    # reset the environment
    env = initialize_environment()
    state = env.reset()
    for i in range(n_steps):
        # choose an action at random
        action = env.action_space.sample()
        env.render()
        new_state, reward, done, info = env.step(action)
        print(f"New State: {new_state}\n"
              f"reward: {reward}\n"
              f"done: {done}\n"
              f"info: {info}\n")
        print("*" * 20)


def value_iteration(env):
    """
    Performs Value Iteration to find the most optimal policy for the
    Tax-v3 environment

    Args:
        env: Taxiv3 Gym environment

    Returns:
        policy: the most optimum policy
    """
    V = dict()
    gamma = 0.9
    state_size = env.observation_space.n
    action_size = env.action_space.n
    policy = dict()
    # initialize the value table randomly
    # initialize the policy randomly
    for x in range(state_size):
        V[x] = -1
        policy[x] = env.action_space.sample()
    # this loop repeats until the change in value function
    # is less than delta
    while True:
        delta = 0
        for state in reversed(range(state_size)):
            old_v_s = V[state]
            best_rewards = -np.inf
            best_action = None
            # for all the actions in current state
            for action in range(action_size):
                # check the reward obtained if we were to perform
                # this action
                prob, new_state, reward, done = env.env.P[state][action][0]
                potential_reward = reward + gamma * V[new_state]
                # select the one that has the best reward
                # and also save the action to the policy
                if potential_reward > best_rewards:
                    best_rewards = potential_reward
                    best_action = action
            policy[state] = best_action
            V[state] = best_rewards
            # terminate if the change is not high
            delta = max(delta, abs(V[state] - old_v_s))
        if delta < 1e-30:
            break
    print(policy)
    print(V)
    return policy


if __name__ == '__main__':
    env = initialize_environment()
    # policy = policy_iteration(env)
    policy = value_iteration(env)
    play(policy, render=True)
