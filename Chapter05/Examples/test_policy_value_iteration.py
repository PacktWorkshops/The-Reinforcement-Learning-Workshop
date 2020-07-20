import import_ipynb
from PolicyandValueIteration_Taxiv3 import policy_iteration, \
    initialize_environment, play, value_iteration


def test_policy_iteration():
    env = initialize_environment()
    policy = policy_iteration(env)
    # will return only if converged


def test_value_iteration():
    env = initialize_environment()
    policy = value_iteration(env)
    # will return only if converged
