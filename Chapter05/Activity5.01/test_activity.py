import import_ipynb
from Activity5_01 import (initialize_environment, policy_iteration, play,
            value_iteration)


def test_policy_iteration():
    env = initialize_environment()
    policy = policy_iteration(env)
    score = play(policy)
    assert 1 == score


def test_value_iteration():
    env = initialize_environment()
    policy = value_iteration(env)
    score = play(policy)
    assert 1 == score
